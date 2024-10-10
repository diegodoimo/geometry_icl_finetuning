from transformers import AutoTokenizer


def get_tokenizer(tokenizer_path=None, model_path=None):

    assert tokenizer_path is not None or model_path is not None
    if tokenizer_path is None:
        tokenizer_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False
    )  # check what happens when true

    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer
