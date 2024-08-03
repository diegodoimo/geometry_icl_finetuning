from datasets import load_dataset
import torch
from functools import partial
from datasets.utils.logging import disable_progress_bar
import sys
import numpy as np
from datasets import load_dataset, load_from_disk
from collections import Counter

rng = np.random.default_rng(42)
disable_progress_bar()

IGNORE_INDEX = -100


def filter_out_long_sequences(tokenized_dataset, max_seq_len):

    tot_examples = tokenized_dataset.num_rows
    processed_dataset = tokenized_dataset.filter(
        lambda example: len(example["input_ids"]) <= max_seq_len
    )
    tot_filtered_examples = processed_dataset.num_rows

    if tot_filtered_examples < tot_examples:
        diff = tot_examples - tot_filtered_examples
        print(
            f"you filter out {diff} examples, {diff/tot_examples*100: .2f}% of the total"
        )
        sys.stdout.flush()


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


# prompt builder
class mmlu_pro_race:
    # num_few_shots = # shots
    # model_name number_istances to remove
    def __init__(
        self,
        dataset_path,
        tokenizer,
        max_seq_len,
        accelerator,
        num_few_shots=0,
        num_processes=1,
        num_samples=None,
        split="test",
        mask_path=None,
        samples_per_subject=None,
        subject=None,
    ):

        self.dataset_path = dataset_path
        self.answers = np.array(["A", "B", "C", "D"])
        self.num_few_shots = num_few_shots
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_processes = num_processes
        self.num_samples = num_samples
        self.accelerator = accelerator
        self.split = split
        self.mask_path = mask_path
        self.samples_per_subject = samples_per_subject
        self.subject = subject

    # ****************************************************
    def construct_question(self, question, choices, answer, include_answer=False):
        # added strip
        prompt = f"{question.strip()}\n"
        for i, choice in enumerate(choices):
            # added strip
            prompt += f"{self.answers[i]}. {choice.strip()}\n"
        # added space to final answers
        prompt += "Answer:"
        if include_answer:
            prompt += f" {self.answers[answer]}\n\n"
        return prompt

    # *********************************************************

    ############

    #  THIS IS THE FUNCTION FOR CONSTRUCTION FEW-SHOT PROMPTS

    ############

    # prompt contruction.buils to operate on list of inputs.
    def construct_prompt(self, batch, tokenizer, dev_set, max_seq_len, num_few_shots):

        prompts = []

        questions = batch["question"]  # list of strings
        subjects = batch["subject"]  # list of strings
        choices = batch["choices"]  # list of list of strings
        answer_indices = np.array(batch["answer"])  # array of integers

        # build a dict of subsets of the dev set with the subject of the batch
        if num_few_shots > 0:
            local_dev_set = {}
            for subject in set(subjects):
                local_dev_set[subject] = dev_set.filter(
                    lambda dev_example, current=subject: dev_example["subject"]
                    == current
                )

        for i in range(len(questions)):
            prompt = f"The following are multiple choice questions (with answers) about{format_subject(subjects[i])}.\n\n"
            current_subject = subjects[i]
            for j in range(num_few_shots):
                shot = local_dev_set[current_subject][j]
                prompt += self.construct_question(
                    shot["question"],
                    shot["choices"],
                    shot["answer"],
                    include_answer=True,
                )
            question = self.construct_question(
                questions[i], choices[i], answer_indices[i]
            )
            prompt += question
            prompts.append(prompt)

        # tokenization part
        tokenized_examples = [
            tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=False,
                add_special_tokens=True,
            ).input_ids.flatten()
            for prompt in prompts
        ]

        # targets are tokenized with space included
        tokenized_labels = [
            tokenizer(
                self.answers[index], return_tensors="pt", add_special_tokens=False
            ).input_ids.flatten()
            for index in answer_indices
        ]

        attention_mask = [
            torch.ones_like(input_ids) for input_ids in tokenized_examples
        ]

        return {
            "prompt": prompts,
            "answers": [self.answers[index] for index in answer_indices],
            "subjects": subjects,
            "input_ids": tokenized_examples,
            "labels": tokenized_labels,
            "attention_mask": attention_mask,
        }

    def construct_prompt_train(
        self, batch, tokenizer, dev_set, max_seq_len, num_few_shots=None
    ):
        # dev_set and few_shots are not used here
        prompts = []
        premises = []

        questions = batch["question"]  # list of strings
        subjects = batch["subject"]  # list of strings
        choices = batch["choices"]  # list of list of strings
        answer_indices = np.array(batch["answer"])  # array of integers

        for i in range(len(questions)):
            question = self.construct_question(
                questions[i],
                choices[i],
                answer_indices[i],
                include_answer=False,
            )
            answer = f" {self.answers[answer_indices[i]]}"
            prompt = question + answer
            prompts.append(prompt)
            premises.append(question)

        # tokenization part
        tokenized_examples = [
            tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=False,
                add_special_tokens=True,
            ).input_ids.flatten()
            for prompt in prompts
        ]

        # tokenized questions
        tokenized_questions = [
            tokenizer(
                question,
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=False,
                add_special_tokens=True,
            ).input_ids.flatten()
            for question in premises
        ]

        # mask out question part
        tokenized_labels = [example.clone() for example in tokenized_examples]

        for i, label_i in enumerate(tokenized_labels):
            label_i[: len(tokenized_questions[i])] = IGNORE_INDEX
            tokenized_labels[i] = label_i

        # double check
        for label in tokenized_labels:
            assert label[-1] != IGNORE_INDEX

        attention_mask = [
            torch.ones_like(input_ids) for input_ids in tokenized_examples
        ]

        return {
            "prompt": prompts,
            "answers": [self.answers[index] for index in answer_indices],
            "subjects": subjects,
            "input_ids": tokenized_examples,
            "labels": tokenized_labels,
            "attention_mask": attention_mask,
        }

    def construct_balanced(self, dataset, samples_per_subject, mask_path):
        assert samples_per_subject is not None or mask_path is not None

        if self.mask_path is not None:
            mask = np.load(f"{self.mask_path}/mask_{samples_per_subject}.npy")
            final = dataset.select(mask)

        else:
            subjects = np.array(dataset["subject"])
            mask = []
            for sub in np.unique(subjects):
                ind = np.nonzero(sub == subjects)[0]
                nsamples = min(samples_per_subject, len(ind))
                chosen = rng.choice(ind, nsamples, replace=False)
                mask.extend(list(np.sort(chosen)))

            mask = np.array(mask)
            final = dataset.select(mask)
        return final

    #####################

    # MAIN FUNCTION BELOW

    #####################

    def construct_dataset(self):

        if self.split == "train":
            assert self.num_few_shots == 0
            dataset = load_from_disk(f"{self.dataset_path}/train")
            if self.samples_per_subject is not None:
                dataset = self.construct_balanced(
                    dataset,
                    mask_path=self.mask_path,
                    samples_per_subject=self.samples_per_subject,
                )

        else:
            dataset = load_from_disk(f"{self.dataset_path}/test")
            if self.subject is not None:
                dataset = dataset.filter(
                    lambda example: example["subject"] in self.subject
                )

        few_shot_dataset = None
        if self.num_few_shots > 0:
            if self.num_few_shots > 5:
                assert False
            few_shot_dataset = load_from_disk(f"{self.dataset_path}/old/dev")

            if self.subject is not None:
                dataset = dataset.filter(
                    lambda example: example["subject"] in self.subject
                )

        # *********************************************************************************
        # contruct prompt and tokenize
        prompt_func = self.construct_prompt
        if self.split == "train":
            prompt_func = self.construct_prompt_train

        encode_function = partial(
            prompt_func,
            tokenizer=self.tokenizer,
            dev_set=few_shot_dataset,
            max_seq_len=self.max_seq_len,
            num_few_shots=self.num_few_shots,
        )
        self.accelerator.print("tokenization started")
        sys.stdout.flush()
        tokenized_dataset = dataset.map(
            encode_function,
            batched=True,
            batch_size=self.num_processes,
            num_proc=self.num_processes,
            load_from_cache_file=False,
        )

        self.accelerator.print("tokenization finished")
        sys.stdout.flush()

        # ************************************************************************
        # remove long sequences
        def sort_by_token_length(example):
            return len(example["input_ids"])

        sorted_indices = sorted(
            range(len(tokenized_dataset)),
            key=lambda i: sort_by_token_length(tokenized_dataset[i]),
            reverse=True,
        )
        longest_sequences = tokenized_dataset.select(sorted_indices[:10])
        longest_sequences.set_format(type="pt")

        tokenized_dataset.set_format(type="pt")
        _ = filter_out_long_sequences(tokenized_dataset, self.max_seq_len)

        def truncate_from_left(example):
            if len(example["input_ids"]) > self.max_seq_len:
                example["input_ids"] = example["input_ids"][-self.max_seq_len :]
                example["attention_mask"] = example["attention_mask"][
                    -self.max_seq_len :
                ]

            return example

        tokenized_dataset = tokenized_dataset.map(truncate_from_left)
        tokenized_dataset.set_format(type="pt")
        _ = filter_out_long_sequences(tokenized_dataset, self.max_seq_len)

        # ******************************
        # add subject weight: used for unbalanced training

        counts = Counter(tokenized_dataset["subject"])
        # sum of the weights must give the number of classes to be consistent with the unifom case
        weights = {
            key: len(tokenized_dataset) / (len(counts) * count)
            for key, count in counts.items()
        }
        tokenized_dataset = tokenized_dataset.add_column(
            "sample_weight", [[weights[sub]] for sub in tokenized_dataset["subject"]]
        )
        tokenized_dataset.set_format(type="pt")

        return tokenized_dataset, longest_sequences
