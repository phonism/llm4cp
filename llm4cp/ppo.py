import torch
import torch.nn.functional as F
import logging
from actor import Actor
from reward import Reward
import deepspeed
import argparse
from config import ActorConfig, RewardConfig, parse_args
from oj import OnlineJudge
from utils import *

def gather_logprobs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)) 
    return log_probs_labels.squeeze(-1)

class ActorCritic(object):
    def __init__(self, args):
        self.args = args
        self.actor = self._init_actor()
        self.ref = self._init_ref()
        self.reward = self._init_reward()
        self.critic = self._init_critic()

    def _init_actor(self):
        actor_config = ActorConfig()
        actor = Actor(actor_config, self.args)
        return actor

    def _init_ref(self):
        actor_config = ActorConfig()
        actor = Actor(actor_config, self.args, is_train=False)
        return actor

    def _init_reward(self):
        reward_config = RewardConfig()
        reward = Reward(reward_config, self.args, is_train=False)
        return reward

    def _init_critic(self):
        reward_config = RewardConfig()
        critic = Reward(reward_config, self.args)
        return critic

class PPO(object):
    def __init__(self, args):
        self.args = args
        self.actor_critic = ActorCritic(args)
        self.actor = self.actor_critic.actor
        self.critic = self.actor_critic.critic
        self.reward = self.actor_critic.reward
        self.ref = self.actor_critic.ref

        self.kl_ctl = 0.02
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

        self.debug = False
        self.oj = OnlineJudge()

    def train(self):
        self.actor.model.train()
        self.critic.model.train()

    def eval(self):
        self.actor.model.eval()
        self.critic.model.eval()
        self.reward.model.eval()
        self.ref.model.eval()

    def save(self):
        save_zero_three_model(self.actor.model, self.actor.tokenizer, self.args.global_rank, "output/rlhf-actor")

    def _generate_sequence(self, prompts):
        with torch.no_grad():
            output, prompt_length = self.actor.generate(prompts, num_beams=10, num_answers=8)
        out_seq = []
        for i in range(len(output)):
            out_seq.append(output[i])
        #out_seq = torch.cat(out_seq, dim=0)
        return out_seq, prompt_length

    def compute_rewards(self, start, logprobs, ref_logprobs, reward_score, action_mask):
        kl_divergence_estimate = -self.kl_ctl * (logprobs - ref_logprobs)
        rewards = kl_divergence_estimate
        ends = start + action_mask[:, start:].sum(1)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
        batch_size = logprobs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]
        return rewards

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns


    def train_rlhf(self, prompts, pid=None):
        #if self.args.global_rank == 0:
            #print(self.actor.model.requires_grad)
        self.eval()
        if self.args.global_rank == 0:
            logging.info("generate start!")
        seq_list, prompt_length = self._generate_sequence(prompts)
        if self.args.global_rank == 0:
            logging.info("generate done!")
        self.train()

        for seq in seq_list:
            ori_seq = seq
            seq = seq.view(1, -1)
            if args.global_rank == 0:
                logging.info("seq.shape" + str(seq.shape))

            pad_token_id = self.actor.tokenizer.pad_token_id
            attention_mask = seq.not_equal(pad_token_id).long()
    
            with torch.no_grad():
                output = self.actor.model(seq, attention_mask=attention_mask)
                output_ref = self.ref.model(seq, attention_mask=attention_mask)
                if self.args.use_real_reward:
                    code_string = self.actor.tokenizer.decode(ori_seq).split("### Response:")[-1].replace("<unk>", "").strip()[:-4]
                    print(code_string)
                    score = oj.score(pid, code_string)
                    print(score)
                    reward_score = torch.tensor([score], device=self.actor.device)
                else:
                    reward_score = self.reward.model(seq, attention_mask).detach()[:, -1]
                values = self.critic.model(seq, attention_mask).detach()[:, :-1]
    
            logits = output.logits
            ref_logits = output_ref.logits
            logprobs = gather_logprobs(logits[:, :-1, :], seq[:, 1:])
            ref_logprobs = gather_logprobs(ref_logits[:, :-1, :], seq[:, 1:])
    
            if args.global_rank == 0 and self.debug:
                print("logprobs", logprobs.detach())
                print("ref_logprobs", ref_logprobs.detach())
                print("logprobs.shape:", logprobs.shape)
                print("attention_mask.shape:", attention_mask.shape)
                print("values.shape", values.shape)
    
            start = prompt_length - 1
            action_mask = attention_mask[:, 1:]
            old_values = values
            with torch.no_grad():
                old_rewards = self.compute_rewards(start, logprobs, ref_logprobs, reward_score, action_mask)
                advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)
    
            batch = {'input_ids': seq, "attention_mask": attention_mask}
            actor_prob = self.actor.model(seq, attention_mask, use_cache=False).logits
            actor_logprobs = gather_logprobs(actor_prob[:, :-1, :], seq[:, 1:])
            if args.global_rank == 0 and self.debug:
                print("action_mask", action_mask[:, start:])
            actor_loss = self.actor_loss_fn(actor_logprobs[:, start:], logprobs[:, start:], advantages, action_mask[:, start:])
    
            if args.global_rank == 0 and self.debug:
                print("old_rewards:", old_rewards.detach())
                print("advantages:", advantages.detach())
                print("returns:", returns.detach())
    
            if args.global_rank == 0:
                logging.info("action_mask:" + str(action_mask.detach()))
                logging.info("old_rewards:" + str(old_rewards.detach()))
                logging.info("actor_loss:" + str(actor_loss.detach().item()))
            self.actor.model.backward(actor_loss)
            self.actor.model.step()
            if args.global_rank == 0:
                logging.info("actor backward done!")

            value = self.critic.model(seq, attention_mask)[:, :-1]
            critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,start:], returns, action_mask[:, start:])
            if self.args.global_rank == 0:
                logging.info("critic_loss:" + str(critic_loss.detach().item()))
    
            self.critic.model.backward(critic_loss)
            self.critic.model.step()
            if self.args.global_rank == 0:
                logging.info("critic backward done!")


    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        values_clipped = torch.clamp(values, old_values - self.cliprange_value, old_values + self.cliprange_value)
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

if __name__ == "__main__":
    logging.info("start!")
    args = parse_args()
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()
        args.global_rank = torch.distributed.get_rank()
        torch.distributed.barrier()
    ppo = PPO(args)

    oj = OnlineJudge()
    for epoch in range(100):
        for pid in sorted(oj.pid_dict):
            problem_statment = oj.pid_dict[pid]["problem_statment"]
            ppo.train_rlhf(problem_statment, pid=pid)
        ppo.save()
