import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_train_ds_config(
        offload, stage=2, enable_hybrid_engine=False, 
        inference_tp_size=1, release_inference_cache=False, 
        pin_parameters=True, tp_gather_partition_size=8):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device,
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": device,
            "pin_memory": True
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_gather_16bit_weights_on_model_save": True,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }
    }


def get_optimizer_grouped_parameters(
        model, weight_decay, no_decay_name_list=["bias", "LayerNorm.weight"]):
    optimizer_grouped_parameters = [ 
            { 
                "params": [
                    p for n, p in model.named_parameters() 
                    if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad) 
                ], 
                "weight_decay": weight_decay,
            }, 
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (any(nd in n for nd in no_decay_name_list) and p.requires_grad) 
                ], 
                "weight_decay": 0.0,
            },
    ]
    return optimizer_grouped_parameters 

def save_zero_three_model(model_ema, tokenizer, global_rank, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
    model_to_save = model_ema.module if hasattr(model_ema, 'module') else model_ema
    model_ema.save_16bit_model(save_dir, WEIGHTS_NAME)
    if global_rank == 0:
        CONFIG_NAME = "config.json"
        output_config_file = os.path.join(save_dir, CONFIG_NAME)
        model_to_save.model.config.to_json_file(output_config_file)
        tokenizer.save_pretrained(save_dir)
