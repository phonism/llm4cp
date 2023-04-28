from utils import *
import time
import copy
import argparse
import json
from torch.utils.data import Dataset, DataLoader
import os
import logging
import math
import sys
import random
from config import ActorConfig, parse_args
from oj import OnlineJudge

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        SchedulerType,
        default_data_collator,
        GenerationConfig,
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


class ActorDataset(Dataset):
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
        prompt = generate_prompt(self.data[idx]["instruction"], self.data[idx]["output"]) + "</s>"
        return (instruct, prompt)

    def __len__(self):
        return len(self.data)

class Actor(object):
    def __init__(self, config, args, is_train=True):
        self.config = config
        self.args = args
        logging.info(args)

        #self.device = torch.device("cuda")
        self.device = torch.device("cuda", args.local_rank)
        self.create_model(is_train=is_train)

    def create_model(self, is_train=True):
        gradient_accumulation_steps = 4
        num_warmup_steps = 0
        self.num_train_epochs = 1
        train_micro_batch_size_per_gpu = 1
        weight_decay = 0.1
        learning_rate = 2e-5
        self.ds_config = get_train_ds_config(offload=self.args.use_offload, stage=self.args.zero_stage)
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

        if self.config.load_dataset:
            train_dataset = ActorDataset(self.config.train_data_path)
            train_sampler = DistributedSampler(train_dataset)
            self.train_dataloader = DataLoader(
                    train_dataset,
                    sampler=train_sampler,
                    batch_size=train_micro_batch_size_per_gpu)
            dataset_size = len(self.train_dataloader)
        else:
            dataset_size = 10000

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.model, weight_decay)
        AdamOptimizer = FusedAdam
        if self.args.use_offload:
            AdamOptimizer = DeepSpeedCPUAdam
        self.optimizer = AdamOptimizer(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.95))

        num_update_steps_per_epoch = math.ceil(dataset_size / gradient_accumulation_steps)
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_train_epochs * num_update_steps_per_epoch)

        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                config=self.ds_config,
                lr_scheduler=self.lr_scheduler,
                dist_init_required=True)
        self.model.gradient_checkpointing_enable()

    def load_model(self):
        dschf = HfDeepSpeedConfig(self.ds_config)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config.model_name_or_path)
        self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
        )
        self.tokenizer.padding_side = "left"  # Allow batched inference
        logging.info("load tokenizer done!")

        self.model =  LlamaForCausalLM.from_pretrained(
                self.config.model_name_or_path)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        #self.model.resize_token_embeddings(int(8 * math.ceil(len(self.tokenizer) / 8.0)))

    def save_model(self):
        model_to_save = self.model.module if hasattr(self.model, "module") else model
        output_dir = "output/actor"
        CONFIG_NAME = "config.json"
        WEIGHTS_NAME = "pytorch_model.bin"
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        save_dict = model_to_save.state_dict()
        torch.save(save_dict, output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        #self.tokenizer.save_vocabulary(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def train(self):
        for epoch in range(self.num_train_epochs):
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                with torch.no_grad():
                    questions = batch[0]
                    query_answer = batch[1]
                    input_tokens_q = self.tokenizer(
                            questions,
                            max_length=2048,
                            return_tensors="pt",
                            truncation=True,
                            padding=False,
                    )
                    input_tokens_qa = self.tokenizer(
                            query_answer,
                            max_length=2048,
                            return_tensors="pt",
                            truncation=True,
                            padding=False,
                    )
                    labels = copy.deepcopy(input_tokens_qa["input_ids"])
                    for label, question in zip(labels, input_tokens_q["input_ids"]):
                        label[:len(question)] = -100
                    input_tokens = {
                            "input_ids": input_tokens_qa["input_ids"].to(self.device),
                            "attention_mask": input_tokens_qa["attention_mask"].to(self.device),
                            "labels": labels.to(self.device),
                    }
                # model with labels will output loss
                outputs = self.model(**input_tokens, use_cache=False)
                loss = outputs.loss
                self.model.backward(loss)
                self.model.step()
                if step % 10 == 0:
                    logging.info("Step:" + str(step) + " loss:" + str(loss.detach().item()))
                if step % 1000 == 0:
                    save_zero_three_model(self.model, self.tokenizer, self.args.global_rank, "output/actor")
                    #self.save_model()
            #self.save_model()
            save_zero_three_model(self.model, self.tokenizer, self.args.global_rank, "output/actor")

    def generate_rlhf(self, prompt):
        prompt = generate_prompt(prompt, "")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            seq = self.model.generate(input_ids, max_length=2048, use_cache=False)
        return seq, input_ids.shape[1]

    def generate(self, prompt, temperature=1, top_p=0.75, 
            top_k=40, num_beams=4, max_new_tokens=1024, readable=False, num_answers=1, **kwargs,):
        prompt = generate_prompt(prompt, "")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                num_return_sequences=num_answers,
                **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
            )
        if readable:
            output_list = []
            for i in range(len(generation_output.sequences)):
                s = generation_output.sequences[i]
                output = self.tokenizer.decode(s).split("### Response:")[-1]
                if num_answers > 1:
                    output = output.replace("<unk>", "")
                output_list.append(output[:-4])
            return output_list
        else:
            output_list = []
            for i in range(len(generation_output.sequences)):
                s = generation_output.sequences[i]
                output_list.append(s)
            return output_list, input_ids.shape[1]

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
        logging.info("init done!")

    actor_config = ActorConfig()
    logging.info("load config done!")
    actor = Actor(actor_config, args)
    actor.train()

def generate():
    args = parse_args()
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()
        args.global_rank = torch.distributed.get_rank()
        torch.distributed.barrier()
        logging.info("init done!")
    actor_config = ActorConfig()
    actor = Actor(actor_config, args)

    oj = OnlineJudge()

    all_tests = 0
    correct_tests = 0
    samples = []
    for pid in sorted(oj.pid_dict):
        one_sample = {}
        problem_statment = oj.pid_dict[pid]["problem_statment"]
        one_sample["instruction"] = problem_statment
        code_string_list = actor.generate(oj.pid_dict[pid]["problem_statment"], readable=True, num_answers=1)
        for code_string in code_string_list:
            one_sample["response"] = code_string
            #a, b = oj.run(pid, code_string)
            one_sample["score"] = oj.score(pid, code_string)
            samples.append(one_sample)
            print(one_sample)

def generate_rlhf():
    args = parse_args()
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()
        args.global_rank = torch.distributed.get_rank()
        torch.distributed.barrier()
        logging.info("init done!")
    actor_config = ActorConfig()
    actor = Actor(actor_config, args)

    oj = OnlineJudge()

    all_tests = 0
    correct_tests = 0
    samples = []
    for pid in sorted(oj.pid_dict):
        one_sample = {}
        problem_statment = oj.pid_dict[pid]["problem_statment"]
        one_sample["instruction"] = problem_statment
        start_time = time.time()
        logging.info(str(args.global_rank) + " start:" + str(start_time))
        _, length = actor.generate(oj.pid_dict[pid]["problem_statment"], num_beams=10, num_answers=10)
        logging.info(str(args.global_rank) + " stop:" + str(time.time() - start_time) + " length:" + str(length))



if __name__ == "__main__":
    #train()
    generate()
    #generate_rlhf()
