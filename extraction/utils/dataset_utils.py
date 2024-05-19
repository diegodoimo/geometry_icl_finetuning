from datasets import load_dataset, concatenate_datasets
import torch
from functools import partial
from datasets.utils.logging import disable_progress_bar
import numpy as np

# from dataset_utils.utils import MMLU_Dataset
import sys
import random
import json
from collections import Counter

disable_progress_bar()

rng = np.random.default_rng(42)


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
    else:
        print(f"there are no sequences longer than {max_seq_len}")
        sys.stdout.flush()


with open("diego/extraction/utils/asset/mmlu_macro_areas.json", "r") as f:
    area_to_subjects = json.load(f)

subject_list = []
for value in area_to_subjects.values():
    subject_list.extend(value)

subject_to_area = {}
for subject in np.unique(subject_list):
    if subject in area_to_subjects["stem"]:

        subject_to_area[subject] = "stem"
    else:
        subject_to_area[subject] = "not_stem"


# prompt builder
class MMLU_Dataset:
    # num_few_shots = # shots
    # model_name number_istences to remove
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        accelerator,
        num_few_shots=0,
        subject=None,
        num_processes=1,
        num_samples=None,
        split="test",
        gibberish=False,
        dummy=False,
        random_subject=False,
        wrong_answers=False,
        sample_questions=False,
        declarative=False,
        aux_few_shot=False,
        only_answer=False,
        only_question=False,
        skip_answer=False,
        skip_choices=False,
        random_order=False,
    ):

        self.dataset = "mmlu"
        self.subject = subject
        if subject is not None:
            self.dataset = f"mmlu:{self.subject}"
        # we add space because the prompt format ends with ":" without a space.
        # comparing the answers in the token space requires this construction.
        self.answers = np.array(["A", "B", "C", "D"])
        self.num_few_shots = num_few_shots
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_processes = num_processes
        self.num_samples = num_samples
        self.accelerator = accelerator
        self.split = split
        self.gibberish = gibberish
        self.dummy = dummy
        self.random_subject = random_subject
        self.wrong_answers = wrong_answers
        self.sample_questions = sample_questions
        self.declarative = declarative
        self.aux_few_shot = aux_few_shot
        self.only_answer = only_answer
        self.only_question = only_question
        self.skip_answer = skip_answer
        self.skip_choices = skip_choices
        self.random_order = random_order

        # self.dummy_examples = self.construct_gibberish_questions(
        #     path="diego/extraction/utils/asset/dummy.txt"
        # )
        # self.gibberish_examples = self.construct_gibberish_questions(
        #     path="diego/extraction/utils/asset/gibberish.txt"
        # )

    def construct_gibberish_questions(self, path):
        # for the moment is 5 shot
        with open(f"{path}", "r", encoding="utf-8") as f:
            lines = f.readlines()

            lines = [line.replace("\\n", "\n")[:-1] for line in lines]

        prompt = ""
        for line in lines:
            prompt += line
        prompt += "\n"

        return prompt

    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def construct_question(
        self,
        question,
        choices,
        answer,
        include_answer=False,
        only_question=False,
        only_answer=False,
        skip_answer=False,
        skip_choices=False,
    ):
        if only_answer:
            # rjust answer template
            prompt = "Answer:"
            prompt += f" {self.answers[answer]}\n\n"

        elif only_question:
            # just question template
            prompt = f"{question.strip()}\n\n"

        elif skip_answer:
            # just question+ options
            prompt = f"{question.strip()}\n\n"
            for i, choice in enumerate(choices):
                # added strip
                prompt += f"{self.answers[i]}. {choice.strip()}\n"
            prompt += "\n"
        elif skip_choices:
            prompt = f"{question.strip()}\n\n"
            prompt += "Answer:"
            prompt += f" {self.answers[answer]}\n\n"
        else:
            # original question + options + answer
            prompt = f"{question.strip()}\n"
            for i, choice in enumerate(choices):
                # added strip
                prompt += f"{self.answers[i]}. {choice.strip()}\n"

            prompt += "Answer:"
            if include_answer:
                if self.wrong_answers:
                    ans_list = list(self.answers)
                    ans_list.remove(self.answers[answer])
                    prompt += f" {random.choice(ans_list)}\n\n"
                else:
                    prompt += f" {self.answers[answer]}\n\n"
        return prompt

    def sample_subject(self, subject):
        # all_areas  = ["stem", "humanities", "other", "social_sciences"]
        all_areas = ["stem", "not_stem"]
        current_area = subject_to_area[subject]

        if current_area == "stem":
            few_shot_area = "not_stem"
        elif current_area == "not_stem":
            few_shot_area = "stem"
        else:
            raise ValueError

        # all_areas.pop(current_area)
        # assert current_area not in all_areas
        # few_shot_area = random.choice(all_areas)
        few_shot_subject = random.choice(area_to_subjects[few_shot_area])
        return few_shot_subject

    def get_few_shot_dataset(
        self,
    ):

        dev_set = load_dataset("cais/mmlu", "all", split="dev")

        val_set = load_dataset("cais/mmlu", "all", split="validation")
        subjects = np.array(val_set["subject"])

        mask = []
        for sub in np.unique(subjects):
            ind = np.nonzero(sub == subjects)[0]
            nsamples = min(8, len(ind))
            chosen = rng.choice(ind, nsamples, replace=False)
            mask.extend(list(np.sort(chosen)))
        mask = np.array(mask)
        val_set_balanced = val_set.select(mask)
        final = concatenate_datasets([dev_set, val_set_balanced])

        # just double check that all is fine
        counts = Counter(final["subject"])
        assert len(np.unique(list(counts.values()))) == 1
        assert np.unique(list(counts.values()))[0] == 13
        self.max_prompt_questions = 13

        return final

    # prompt contruction.buils to operate on list of inputs.
    def construct_prompt(
        self, batch, tokenizer, dev_set, max_seq_len, num_few_shots, aux_few_shot=None
    ):
        prompts = []

        questions = batch["question"]  # list of strings
        subjects = batch["subject"]  # list of strings
        choices = batch["choices"]  # list of list of strings
        answer_indices = np.array(batch["answer"])  # array of integers

        # build a dict of subsets of the dev set with the subject of the batch
        if num_few_shots > 0:
            local_dev_set = {}
            local_aux_set = {}
            for subject in set(subjects):
                prompt_subject = subject
                if self.declarative:

                    local_dev_set[subject] = dev_set[subject]

                    if aux_few_shot is not None:
                        local_aux_set[subject] = aux_few_shot.filter(
                            lambda dev_example: dev_example["subject"]
                            == prompt_subject,
                        )
                else:
                    if self.random_subject:
                        prompt_subject = self.sample_subject(subject)

                    local_dev_set[subject] = dev_set.filter(
                        lambda dev_example: dev_example["subject"] == prompt_subject,
                    )

        for i, question in enumerate(questions):

            if self.dummy:
                prompt = "The following are vorpal borogoves (with gyres) about the frumious bandersnatch.\n\n"
                prompt += self.construct_gibberish_questions(
                    path="diego/extraction/utils/asset/dummy.txt"
                )
            elif self.gibberish:
                prompt = "Zorpulika blivikwak bakki (floopz wiz zorps) ombli bla.\n\n"
                prompt += self.construct_gibberish_questions(
                    path="diego/extraction/utils/asset/gibberish.txt"
                )
            elif self.declarative:
                prompt = f"The following are statements about{self.format_subject(subjects[i])}.\n\n"

                current_subject = subjects[i]

                num_few_shots_tmp = num_few_shots
                if num_few_shots > len(local_dev_set[current_subject]):
                    num_few_shots_tmp = len(local_dev_set[current_subject])
                    assert (
                        num_few_shots_tmp == self.subject_count[current_subject] + 5
                    ), (
                        current_subject,
                        num_few_shots_tmp,
                        self.subject_count[current_subject] + 5,
                    )

                # indices = rng.permutation(num_few_shots)
                indices = np.arange(num_few_shots_tmp)
                for j in indices:
                    shot = local_dev_set[current_subject][int(j)]
                    prompt += f"{shot}\n\n"

            else:
                prompt = f"The following are multiple choice questions (with answers) about{self.format_subject(subjects[i])}.\n\n"

                current_subject = subjects[i]
                indices = np.arange(num_few_shots)
                if self.random_order:
                    indices = rng.permutation(num_few_shots)

                if self.sample_questions:
                    indices = rng.choice(
                        self.max_prompt_questions, num_few_shots, replace=False
                    )

                for j in indices:
                    shot = local_dev_set[current_subject][int(j)]
                    prompt += self.construct_question(
                        shot["question"],
                        shot["choices"],
                        shot["answer"],
                        include_answer=True,
                        only_answer=self.only_answer,
                        only_question=self.only_question,
                        skip_answer=self.skip_answer,
                        skip_choices=self.skip_choices,
                    )

            question = self.construct_question(
                questions[i], choices[i], answer_indices[i]
            )

            if self.declarative:
                if aux_few_shot is not None:
                    shot = local_aux_set[current_subject][0]
                    prompt += self.construct_question(
                        shot["question"],
                        shot["choices"],
                        shot["answer"],
                        include_answer=True,
                    )
                # prompt += "\n\nThe answer to the following question must be one of the options: A B C or D.\n"
                prompt += f"The following is a multiple choice question (with answers) about{self.format_subject(subjects[i])}. The question ends with Answer: after which, you must reply with one of following letters: A B C D .\n\n"
                prompt += question

            else:
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

    def construct_dataset(self):
        """
        Construct the request instances for the scenario
        """
        # removed trust remote code
        self.accelerator.print("loading dataset")
        split = self.split
        if self.num_samples is not None:
            split = f"test[:{self.num_samples}]"
        if self.subject is not None:
            dataset = load_dataset("cais/mmlu", self.subject, split=split)
        else:
            dataset = load_dataset("cais/mmlu", "all", split=split)

        # few_shot_dataset = None
        # if (
        #     self.num_few_shots > 0
        #     and self.num_few_shots <= 5
        #     and not self.sample_questions
        # ):
        #     few_shot_dataset = load_dataset("cais/mmlu", "all", split="dev")
        # elif self.num_few_shots > 5 or self.sample_questions:
        #     assert self.split != "validation"
        #     if self.sample_questions:
        #         few_shot_dataset = self.get_few_shot_dataset()
        #     else:
        #         few_shot_dataset = load_dataset(
        #             "cais/mmlu", "all", split="dev+validation"
        #         )

        aux_few_shot = None
        if self.declarative:
            assert self.num_few_shots > 0

            with open(f"diego/extraction/utils/mmlu_declarative_dev.json", "r") as f:
                few_shot_dataset_dev = json.load(f)

            with open(f"diego/extraction/utils/mmlu_declarative_val.json", "r") as f:
                few_shot_dataset_val = json.load(f)

            few_shot_dataset = {key: [] for key in few_shot_dataset_dev.keys()}

            for key, val in few_shot_dataset_dev.items():
                few_shot_dataset[key].extend(val)

            for key, val in few_shot_dataset_val.items():
                few_shot_dataset[key].extend(val)

            # check that we have all the subjects
            with open("diego/extraction/utils/mmlu_subject_val15.json", "r") as f:
                subjects_count = json.load(f)
            self.subject_count = subjects_count
            for key, val in few_shot_dataset.items():
                assert len(val) == subjects_count[key] + 5, (
                    key,
                    len(val),
                    subjects_count[key] + 5,
                )

            if self.aux_few_shot:
                aux_few_shot = load_dataset("cais/mmlu", "all", split="dev")

        elif self.sample_questions:
            assert self.num_few_shots > 0
            assert self.split != "validation"
            few_shot_dataset = self.get_few_shot_dataset()

        else:
            few_shot_dataset = None
            if self.num_few_shots > 0 and self.num_few_shots <= 5:
                few_shot_dataset = load_dataset("cais/mmlu", "all", split="dev")

            elif self.num_few_shots > 5:
                assert self.split != "validation"
                few_shot_dataset = load_dataset(
                    "cais/mmlu", "all", split="dev+validation"
                )

        encode_function = partial(
            self.construct_prompt,
            tokenizer=self.tokenizer,
            dev_set=few_shot_dataset,
            max_seq_len=self.max_seq_len,
            num_few_shots=self.num_few_shots,
            aux_few_shot=aux_few_shot,
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

        self.accelerator.print("checking max_seq_len")
        sys.stdout.flush()

        def sort_by_token_length(example):
            return len(example["input_ids"])

        sorted_indices = sorted(
            range(len(tokenized_dataset)),
            key=lambda i: sort_by_token_length(tokenized_dataset[i]),
            reverse=True,
        )
        longest_sequences = tokenized_dataset.select(sorted_indices[:10])
        longest_sequences.set_format(type="pt")

        # remove examples loger than max seq len maybe not necessary at all
        # list of list is made list of tensors
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
        self.accelerator.print("after_truncation")
        tokenized_dataset.set_format(type="pt")
        _ = filter_out_long_sequences(tokenized_dataset, self.max_seq_len)

        return tokenized_dataset, longest_sequences
