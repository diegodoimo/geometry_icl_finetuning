import warnings
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
import sys


def get_model(model_name_or_path, precision, low_cpu_mem_usage, accelerator):

    if model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
        accelerator.print("model_loading started. \n\n")
        sys.stdout.flush()
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=precision,
            use_flash_attention_2=False,
        )

    else:
        warnings.warn("Using a fake llama for debugging\n", stacklevel=2)
        config = LlamaConfig()
        config.intermediate_size = 1000
        config.num_hidden_layers = 3
        config.num_attention_heads = 2
        config.num_key_value_heads = 2
        config.hidden_size = 500
        model = AutoModelForCausalLM.from_config(config)

    accelerator.print("model loading finished. \n\n")
    sys.stdout.flush()

    return model
