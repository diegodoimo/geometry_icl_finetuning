from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Sequence, Dict
import torch
from dataclasses import dataclass
from typing import Sequence, Dict

IGNORE_INDEX = -100


def get_dataloader(
    dataset,
    batch_size,
    pad_token_id,
    world_size=1,
    sampler=None,
    collate_fn=None,
    shuffle=False,
    drop_last=True,
    num_processes=1,
    return_sampler=False,
    weight_samples=False,
):

    if weight_samples:
        collate_fn = WeightedDataCollatorForCausalLM(pad_token_id=pad_token_id)
    elif collate_fn is None:
        collate_fn = DataCollatorForCausalLM(pad_token_id=pad_token_id)

    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_processes,
        sampler=sampler,
        pin_memory=True,
    )

    if return_sampler:
        return dataloader, sampler
    return dataloader


@dataclass
class DataCollatorForCausalLM:
    """Collate examples for supervised fine-tuning."""

    pad_token_id: int
    # max_seq_len: int

    # check if we can set padding value in labels == eos_token_id_directly (as the attention mask should take into account the gradient masking)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # in the structure of open-instruct the instances are already tensors, and already take into account max_seq_len

        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


@dataclass
class DataCollatorForCausalLMEval:
    """Collate examples for supervised fine-tuning."""

    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # in the structure of open-instruct the instances are already tensors, and already take into account max_seq_len

        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]
        subjects = [instance["subjects"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            subjects=subjects,
        )


@dataclass
class WeightedDataCollatorForCausalLM:
    """Collate examples for supervised fine-tuning."""

    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]
        sample_weight = [instance["sample_weight"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        weight = torch.nn.utils.rnn.pad_sequence(
            sample_weight, batch_first=True, padding_value=self.pad_token_id
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            weight=weight,
        )
