from collections import defaultdict
import sys
import numpy as np


def get_embdims(model, dataloader, target_layers):
    embdims = defaultdict(lambda: None)
    dtypes = defaultdict(lambda: None)

    def get_hook(name, embdims):
        def hook_fn(module, input, output):
            try:
                embdims[name] = output.shape[-1]
                dtypes[name] = output.dtype
            except:
                embdims[name] = output[0].shape[-1]
                dtypes[name] = output[0].dtype

        return hook_fn

    handles = {}
    for name, module in model.named_modules():
        if name in target_layers:
            handles[name] = module.register_forward_hook(get_hook(name, embdims))

    batch = next(iter(dataloader))
    sys.stdout.flush()
    _ = model(batch["input_ids"].to("cuda"))

    for name, module in model.named_modules():
        if name in target_layers:
            handles[name].remove()

    assert len(embdims) == len(target_layers)
    return embdims, dtypes


def compute_accuracy(predictions, answers, subjects=None):

    # we remove spaces in from of the letters
    accuracy = {}
    tot_ans = len(predictions)
    num_correct = 0
    for pred, ans in zip(predictions, answers):
        if pred == ans:
            num_correct += 1
    accuracy["micro"] = num_correct / tot_ans

    if subjects is not None:
        acc_subj = {}
        for subject in np.unique(subjects):
            mask = subject == subjects
            pred_tmp = predictions[mask]
            ans_tmp = answers[mask]

            tot_ans = len(ans_tmp)
            num_correct = 0
            for pred, ans in zip(pred_tmp, ans_tmp):
                if pred == ans:
                    num_correct += 1
            acc_tmp = num_correct / tot_ans

            acc_subj[subject] = acc_tmp

    accuracy["subjects"] = acc_subj
    accuracy["macro"] = np.mean(list(acc_subj.values()))

    return accuracy


def measure_performance(extr_act, dataloader, tokenizer, accelerator):

    predictions = extr_act.predictions  # tokens
    constrained_predictions = extr_act.constrained_predictions  # tokens
    answers = dataloader.dataset["answers"]  # letters
    subjects = dataloader.dataset["subjects"]

    answers = np.array([ans.strip() for ans in answers])
    predictions = np.array([tokenizer.decode(pred).strip() for pred in predictions])
    constrained_predictions = np.array(
        [tokenizer.decode(pred).strip() for pred in constrained_predictions]
    )

    acc_pred = compute_accuracy(
        predictions,
        answers[: len(predictions)],
        np.array(subjects[: len(predictions)]),
    )
    acc_constrained = compute_accuracy(
        constrained_predictions,
        answers[: len(constrained_predictions)],
        np.array(subjects[: len(predictions)]),
    )
    accelerator.print("exact_match constrained:", acc_constrained["macro"])
    accelerator.print("exact_match macro:", acc_pred["macro"])
    accelerator.print("exact_match micro:", acc_pred["micro"])
    for subject, acc in acc_pred["subjects"].items():
        accelerator.print(f"{subject}: {acc:.3f}\n")

    statistics = {
        "subjects": dataloader.dataset["subjects"],
        "answers": answers,
        "predictions": predictions,
        "contrained_predictions": constrained_predictions,
        "accuracy": acc_pred,
        "constrained_accuracy": acc_constrained,
    }
    return statistics


def remove_duplicates_func(act, accelerator):

    act, idx, inverse = np.unique(act, axis=0, return_index=True, return_inverse=True)
    accelerator.print(len(idx), len(inverse))
    save_backward_indices=False
    if len(idx) == len(inverse):
        # if no overlapping data has been found return the original ordred array
        assert len(np.unique(inverse)) == len(inverse)
        act = act[inverse]
    else:
        save_backward_indices = True
    accelerator.print(f"unique_samples = {len(idx)}")
    accelerator.print(f"num_duplicates = {len(inverse)-len(idx)}")

    return act, save_backward_indices, inverse
