import time
import torch
import numpy as np
from collections import defaultdict
import torch.distributed as dist
import psutil
import sys

rng = np.random.default_rng(42)
# ***************************************************


class extract_activations:
    def __init__(
        self,
        accelerator,
        model,
        dataloader,
        target_layers,
        embdim,
        dtypes,
        use_last_token=False,
        print_every=100,
        prompt_search=False,
        time_stamp=None,
    ):
        self.accelerator = accelerator
        self.model = model
        # embedding size
        self.embdim = embdim
        # whether to compute the id on the last token /class_token
        self.use_last_token = use_last_token
        self.print_every = print_every
        self.target_layers = target_layers

        self.micro_batch_size = dataloader.batch_size
        self.nbatches = len(dataloader)
        self.world_size = self.accelerator.num_processes

        # number of samples to collect (e.g. 10k) for llama 70B we remove the last 2 samples
        self.nsamples = self.nbatches * self.micro_batch_size * self.world_size

        self.rank = self.accelerator.process_index
        self.global_batch_size = self.world_size * self.micro_batch_size
        self.hidden_size = 0
        self.prompt_search = prompt_search

        print("rank: nsamples", self.rank, self.nsamples)
        print("rank: nbatches", self.rank, self.nbatches)
        sys.stdout.flush()
        if self.rank == 0:
            self.accelerator.print(
                "before hidden states RAM Used (GB):",
                psutil.virtual_memory()[3] / 10**9,
            )
        if prompt_search == False:
            self.init_hidden_states(target_layers, dtypes=dtypes)
            self.init_hooks(target_layers)

    def init_hidden_states(self, target_layers, dtypes):
        # dict containing the representations extracted in a sigle forward pass
        self.hidden_states_tmp = defaultdict(lambda: None)

        # dict storing the all the representations
        if self.accelerator.is_main_process:
            self.hidden_states = {}
            for name in target_layers:
                self.hidden_states[name] = torch.zeros(
                    (self.nsamples, self.embdim[name]), dtype=dtypes[name]
                )

    def init_hooks(self, target_layers):
        for name, module in self.model.named_modules():
            if name in target_layers:
                module.register_forward_hook(
                    self._get_hook(name, self.hidden_states_tmp)
                )

    def _get_hook(self, name, hidden_states):
        if self.world_size > 1:

            def hook_fn(module, input, output):
                hidden_states[name] = output

        else:

            def hook_fn(module, input, output):
                hidden_states[name] = output.cpu()

        return hook_fn

    def all_gather_logits(self, logits, targets, seq_len):

        _, _, embdim = logits.shape
        if self.world_size > 1:
            # gather the logits to rank 0
            logit_list = [
                torch.zeros((1, embdim), device="cuda", dtype=logits.dtype)
                for _ in range(self.world_size)
            ]
            target_list = [
                torch.zeros_like(targets, device="cuda", dtype=targets.dtype)
                for _ in range(self.world_size)
            ]
            dist.all_gather(logit_list, logits[:, seq_len[0] - 1, :])
            dist.all_gather(target_list, targets)
            logits = torch.cat(logit_list, dim=0)
            targets = torch.cat(target_list, dim=0)
        else:
            assert logits.shape[0] == seq_len.shape[0]
            logits = logits[torch.arange(logits.shape[0]), seq_len - 1, :]

        return logits, targets

    def gather_logits(self, logits, seq_len, targets):

        _, _, embdim = logits.shape
        if self.world_size > 1:

            assert seq_len.shape[0] == 1, "batch_size must be 1 when world size >1 "
            if self.rank == 0:
                # gather the logits to rank 0
                logit_list = [
                    torch.zeros((1, embdim), device="cuda", dtype=logits.dtype)
                    for _ in range(self.world_size)
                ]
                target_list = [
                    torch.zeros_like(targets, device="cuda", dtype=targets.dtype)
                    for _ in range(self.world_size)
                ]
                dist.gather(logits[:, seq_len[0] - 1, :], logit_list, dst=0)
                dist.gather(targets, target_list, dst=0)

                logits = torch.cat(logit_list, dim=0)
                targets = torch.cat(target_list, dim=0)
            else:
                dist.gather(logits[:, seq_len[0] - 1, :], dst=0)
                dist.gather(targets, dst=0)
        else:
            assert seq_len.shape[0] == logits.shape[0]
            logits = logits[torch.arange(logits.shape[0]), seq_len - 1, :]

        return logits, targets

    def _gather_and_update_fsdp(self, mask, is_last_batch):
        # batch size ==  1 we handle just this setup for world size > 1
        assert mask.shape[0] == 1

        # all gather the sequence lengths from all ranks
        seq_len = torch.sum(mask, dim=1)  # 1x 1 tensor
        seq_len_list = [torch.zeros_like(seq_len) for _ in range(self.world_size)]
        dist.all_gather(seq_len_list, seq_len)
        max_size = max(seq_len_list).item()
        size_diff = max_size - seq_len.item()

        for _, (name, hidden_state) in enumerate(self.hidden_states_tmp.items()):
            _, _, embdim = hidden_state.shape

            # pad the activations in all ranks to be of the shape 1 x max_seq_len x embd
            if size_diff > 0:
                padding = torch.zeros(
                    (1, size_diff, embdim), device="cuda", dtype=hidden_state.dtype
                )
                hidden_state = torch.cat((hidden_state, padding), dim=1)

            if self.rank == 0:
                # gather the activations to rank 0
                states_list = [
                    torch.zeros(
                        (1, max_size, embdim), device="cuda", dtype=hidden_state.dtype
                    )
                    for _ in range(self.world_size)
                ]
                dist.gather(hidden_state, states_list, dst=0)

                # move to cpu, remove padding, and update hidden states
                num_current_tokens = self._update_hidden_state_fsdp(
                    states_list, seq_len_list, name, is_last_batch
                )
            else:
                dist.gather(hidden_state, dst=0)

            dist.barrier()
            del hidden_state

        if self.rank == 0:
            self.hidden_size += num_current_tokens

        return torch.cat(seq_len_list, dim=0)

    def _update_hidden_state_fsdp(self, states_list, seq_len_list, name, is_last_batch):
        if self.use_last_token:
            act_tmp = torch.cat(
                [
                    state.squeeze()[seq_len.item() - 1 : seq_len.item()]
                    for state, seq_len in zip(states_list, seq_len_list)
                ],
                dim=0,
            ).cpu()
            assert act_tmp.shape[0] == self.world_size
            num_current_tokens = act_tmp.shape[0]
        else:
            act_tmp = torch.cat(
                [
                    state.squeeze()[: seq_len.item()].mean(dim=0, keepdims=True)
                    for state, seq_len in zip(states_list, seq_len_list)
                ],
                dim=0,
            ).cpu()
            assert act_tmp.shape[0] == self.world_size
            num_current_tokens = act_tmp.shape[0]

        if is_last_batch:

            self.hidden_states[name][self.hidden_size :] = act_tmp

        else:
            self.hidden_states[name][
                self.hidden_size : self.hidden_size + num_current_tokens
            ] = act_tmp

        return num_current_tokens

    def _update_hidden_state(self, mask, is_last_batch):
        seq_len = torch.sum(mask, dim=1)
        mask = mask.unsqueeze(-1)
        for i, (name, activations) in enumerate(self.hidden_states_tmp.items()):
            if self.use_last_token:
                batch_size = seq_len.shape[0]
                act_tmp = activations[
                    torch.arange(batch_size), torch.tensor(seq_len) - 1
                ]
            else:
                denom = torch.sum(mask, dim=1)  # batch x 1
                # act_tmp -> batch x seq_len x embed
                # mask -> batch x seq_len x 1
                # denom -> batch x 1
                act_tmp = torch.sum(activations * mask, dim=1) / denom

            num_current_tokens = act_tmp.shape[0]
            if is_last_batch:

                self.hidden_states[name][self.hidden_size :] = act_tmp
            else:
                self.hidden_states[name][
                    self.hidden_size : self.hidden_size + num_current_tokens
                ] = act_tmp

        self.hidden_size += num_current_tokens
        return seq_len

    @torch.inference_mode()
    def extract(self, dataloader, tokenizer):
        start = time.time()
        is_last_batch = False
        choices = ["A", "B", "C", "D"]

        self.predictions = []
        self.constrained_predictions = []
        self.targets = []

        logit_list, batch_list = [], []
        # entries in the vocabulary space restriced to the 4 output options.
        # space is added (see prompt construction)

        candidate_token_ids = tokenizer(
            choices, add_special_tokens=False, return_tensors="pt"
        ).input_ids.flatten()

        for i, data in enumerate(dataloader):
            if (i + 1) == self.nbatches:
                is_last_batch = True

            mask = data["attention_mask"] != 0
            mask = mask.to("cuda")
            batch = data["input_ids"].to("cuda")
            targets = data["labels"].to("cuda")

            outputs = self.model(batch)
            if not self.prompt_search:
                if self.world_size > 1:
                    seq_len = self._gather_and_update_fsdp(mask, is_last_batch)

                else:
                    seq_len = self._update_hidden_state(mask.cpu(), is_last_batch)

            seq_len = torch.sum(mask, dim=1)
            logits, targets = self.all_gather_logits(outputs.logits, targets, seq_len)
            logits, targets = logits.cpu(), targets.cpu()

            if self.rank == 0:
                logits_targets = logits[:, candidate_token_ids]
                constrained_prediction_batch = candidate_token_ids[
                    torch.argmax(logits_targets, dim=-1)
                ]

                self.constrained_predictions += (
                    constrained_prediction_batch.cpu().tolist()
                )
                self.targets += targets.cpu().tolist()

                # unconstrained predictions and targets
                self.predictions += torch.argmax(logits, dim=-1).cpu().tolist()

                if (i + 1) % (self.print_every // self.global_batch_size) == 0:

                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.max_memory_allocated() / 2**30
                    reserved = torch.cuda.max_memory_reserved() / 2**30
                    end = time.time()
                    self.accelerator.print(
                        f"{(i+1)*self.global_batch_size/1000}k data, \
                        batch {i+1}/{self.nbatches}, \
                        tot_time: {(end-start)/60: .3f}min \
                        mem alloc/reserved {allocated: .2f}{reserved: .2f}"
                    )
                    sys.stdout.flush()

        self.predictions = torch.tensor(self.predictions)
        self.constrained_predictions = torch.tensor(self.constrained_predictions)
