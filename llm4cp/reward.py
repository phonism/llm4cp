from utils import (
        get_train_ds_config,
        get_optimizer_grouped_parameters,
        save_zero_three_model,
)
import copy
import argparse
import json
from torch.utils.data import Dataset, DataLoader
import os
import logging
import math
import sys
import random
from config import RewardConfig, parse_args
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


from transformers import (
        PreTrainedModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        SchedulerType,
        default_data_collator,
        get_scheduler,
        LlamaForCausalLM, 
        LlamaTokenizer
)
from transformers.deepspeed import HfDeepSpeedConfig

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_prompt(instruction, response=""):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""


class RewardDataset(Dataset):
    """Dataset class for the actor model
    read a json file with the following format:
    [
        {
            "user_input": "...",
            "completion": "...",
            "score": ...
        },
        ...
    ]
    Where:
        user_input: the initial input of the user
        completion: the completion generated by the model
        score: the score given by the user to the completion (or by the LLM)
    """

    def __init__(self, path: str) -> None:
        logging.info(f"Loading dataset from {path}")
        with open(path, "r") as f:
            self.data = list(json.load(f))
            random.shuffle(self.data)
        logging.info(f"Loaded {len(self.data)} samples")

    def __getitem__(self, idx: int):
        instruct = generate_prompt(self.data[idx]["instruction"], "")
        prompt = generate_prompt(self.data[idx]["instruction"], self.data[idx]["response"])
        return (instruct, prompt, float(self.data[idx]["score"]))

    def __len__(self):
        return len(self.data)

class RewardModel(torch.nn.Module):
    def __init__(self, config):
        super(RewardModel, self).__init__()

        self.config = config

        self.tokenizer = LlamaTokenizer.from_pretrained(self.config.model_name_or_path)
        self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
        )
        self.tokenizer.padding_side = "left"  # Allow batched inference
        logging.info("load tokenizer done!")

        self.model =  LlamaForCausalLM.from_pretrained(
                self.config.model_name_or_path, )
                    #torch_dtype=torch.float16,
                    #device_map="auto")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        self.head = torch.nn.Sequential(
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1),
            Rearrange("... 1 -> ..."),
        )
    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def forward(self, sequences, sequences_mask):
        model_output = self.model(sequences, sequences_mask, use_cache=False, output_hidden_states=True)
        outputs = self.head(model_output.hidden_states[-1])
        return outputs

class Reward(object):
    def __init__(self, config, args, is_train=True):
        self.config = config
        self.args = args
        self.device = torch.device("cuda", args.local_rank)
        self.create_model(is_train=is_train)

    def create_model(self, is_train=True, laod_from_pretrain=True):
        gradient_accumulation_steps = 4
        num_warmup_steps = 0
        self.num_train_epochs = 1
        train_micro_batch_size_per_gpu = 1
        weight_decay = 0.1
        learning_rate = 2e-5

        self.ds_config = get_train_ds_config(offload=self.args.use_offload, stage=3)
        self.ds_config["train_micro_batch_size_per_gpu"] = train_micro_batch_size_per_gpu
        self.ds_config["train_batch_size"] = train_micro_batch_size_per_gpu * gradient_accumulation_steps * 4

        self.load_model()
        logging.info("load model done!")

        if is_train is False:
            self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=self.model,
                config=self.ds_config,
                dist_init_required=True)
            return

        dschf = HfDeepSpeedConfig(self.ds_config)

        train_dataset = RewardDataset(self.config.train_data_path)
        train_sampler = DistributedSampler(train_dataset)
        self.train_dataloader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=train_micro_batch_size_per_gpu)
        
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.model, weight_decay)
        AdamOptimizer = FusedAdam
        if self.args.use_offload:
            AdamOptimizer = DeepSpeedCPUAdam
        self.optimizer = AdamOptimizer(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.95))

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / gradient_accumulation_steps)
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_train_epochs * num_update_steps_per_epoch)

        self.loss_function = torch.nn.MSELoss()

        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                config=self.ds_config,
                lr_scheduler=self.lr_scheduler,
                dist_init_required=True)
        self.model.gradient_checkpointing_enable()

    def load_model(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config.model_name_or_path)
        self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
        )
        self.tokenizer.padding_side = "left"  # Allow batched inference
        logging.info("load tokenizer done!")
        self.model = RewardModel(self.config)
        if self.config.load_from_finetune:
            self.model.load_state_dict(torch.load(self.config.finetune_model_file, map_location='cpu'))

    def save_model(self):
        model_to_save = self.model.module if hasattr(self.model, "module") else model
        output_dir = "output/reward"
        CONFIG_NAME = "config.json"
        WEIGHTS_NAME = "pytorch_model.bin"
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        save_dict = model_to_save.state_dict()
        torch.save(save_dict, output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(output_dir)

    def train(self):
        for epoch in range(self.num_train_epochs):
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                with torch.no_grad():
                    query_answer = batch[1]
                    scores = batch[2]
                    input_tokens_qa = self.tokenizer(
                            query_answer,
                            max_length=2048,
                            return_tensors="pt",
                            truncation=True,
                            padding=False,
                    )
                    input_tokens = {
                            "input_ids": input_tokens_qa["input_ids"].to(self.device),
                            "attention_mask": input_tokens_qa["attention_mask"].to(self.device),
                    }
                    labels = torch.as_tensor(
                            scores, dtype=torch.float16, device=self.device
                    )
                # model with labels will output loss
                outputs = self.model(input_tokens["input_ids"], input_tokens["attention_mask"])[:, -1]
                #logging.info(outputs.detach().item(), labels.detach().item())
                loss = self.loss_function(outputs, labels)
                self.model.backward(loss)
                self.model.step()
                if step % 10 == 0:
                    logging.info("Rank:" + str(self.args.global_rank) + " Step:" + str(step) + " loss:" + str(loss.detach().item()))
                if step % 1000 == 0:
                    save_zero_three_model(self.model, self.args.global_rank, "output/reward")
                    #self.save_model()
            #self.save_model()
            save_zero_three_model(self.model, self.tokenizer, self.args.global_rank, "output/reward")

def train():
    #torch.cuda.set_device(args.local_rank)
    #deepspeed.init_distributed()
    #torch.distributed.barrier()
    #local_rank = torch.distributed.get_rank()
    #device = torch.device("cuda", local_rank)
    args = parse_args()
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()
        args.global_rank = torch.distributed.get_rank()
        torch.distributed.barrier()

    reward_config = RewardConfig()
    reward = Reward(reward_config, args)
    reward.train()


if __name__ == "__main__":
    train()

