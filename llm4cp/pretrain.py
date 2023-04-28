from torch.utils.data import Dataset, DataLoader
from config import PretrainConfig
import math
import os
import sys
import logging
from utils import *
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


class PretrainDataset(Dataset):
    def __init__(self, path: str, config) -> None:
        logging.info(f"Loading dataset from {path}")
        base_dir = ["../../dataset/cp-algorithms/src/", "../../dataset/OI-wiki/docs/", "../../dataset/usaco-guide/content/"]
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name_or_path)
        self.data = []
        for bd in base_dir:
            for fl in os.listdir(bd):
                if not os.path.isdir(bd + "/" + fl):
                    continue
                for fn in os.listdir(bd + "/" + fl):
                    if fn.find(".md") == -1:
                        continue
                    file_content = open(bd + "/" + fl + "/" + fn).read()
                    out = tokenizer(file_content)
                    input_ids = out["input_ids"]
                    for i in range(math.ceil(len(input_ids) / config.max_length)):
                        self.data.append(input_ids[i * config.max_length : (i + 1) * config.max_length])
                #print(file_content)
        print(len(self.data))
        logging.info(f"Loaded {len(self.data)} samples")

    def __getitem__(self, idx: int):
        instruct = generate_prompt(self.data[idx]["instruction"], "")
        prompt = generate_prompt(self.data[idx]["instruction"], self.data[idx]["output"]) + "</s>"
        return (instruct, prompt)

    def __len__(self):
        return len(self.data)

pretrain_config = PretrainConfig()
dataset = PretrainDataset("..", pretrain_config)
