import torch
import math
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(model, args):

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


def get_scheduler(optimizer, args):

    if args.warmup_steps is None and args.warmup_ratio is None:
        warmup_steps = 0
    elif args.warmup_steps is None:
        warmup_steps = int(args.warmup_ratio * args.max_train_steps)
    warmup_steps = int(min(warmup_steps, 3))

    scheduler_func = lambda x: min(
        args.lr_min_fact + (1 - args.lr_min_fact) * min(x, warmup_steps) / warmup_steps,
        args.lr_min_fact
        + 0.5
        * (1 - args.lr_min_fact)
        * (
            1
            + math.cos(
                max(0, x - warmup_steps)
                / (args.max_train_steps - warmup_steps)
                * math.pi
            )
        ),
    )
    scheduler = LambdaLR(optimizer, lambda x: scheduler_func(x))
    return scheduler, warmup_steps
