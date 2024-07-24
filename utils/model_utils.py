import sys
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers import LlamaConfig
import warnings

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def get_model_hf(
    model_name_or_path,
    precision,
    low_cpu_mem_usage,
    accelerator,
    use_flash_attention_2=False,
    activation_checkpointing=False,
):

    if model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
        accelerator.print("model_loading started. \n\n")
        sys.stdout.flush()
        if activation_checkpointing:
            config.use_cache = False
            # by default is true: about use cache
            # https://github.com/huggingface/transformers/issues/28499
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=precision,
            use_flash_attention_2=use_flash_attention_2,
        )
        if activation_checkpointing:
            assert model.config.use_cache == False

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
