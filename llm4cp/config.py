import argparse
import deepspeed

class ActorConfig(object):
    model_name_or_path = ""
    train_data_path = ""
    load_dataset = True

class RewardConfig(object):
    model_name_or_path = "../../../models/llama-7b/"
    #model_name_or_path = "./output/reward"
    train_data_path = "./data/reward_train_all.json"
    load_from_finetune = True
    finetune_model_file = "./output/reward-old/pytorch_model.bin"

class PretrainConfig(object):
    model_name_or_path = "../../../models/llama-7b/"
    max_length = 2048

def parse_args():
    parser = argparse.ArgumentParser(
            description="")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable HF gradient checkpointing for model.")
    parser.add_argument("--use_offload", type=bool, default=False, help="")
    parser.add_argument("--zero_stage", type=int, default=3, help="")
    parser.add_argument("--use_real_reward", type=bool, default=False, help="")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

