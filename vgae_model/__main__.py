"""CLI interface for training/evaluating the VGAE models."""

from vgae_model.utils import pretrain, finetune, test, get_hparams, optimize_lr
import lightning as pl
import torch
import argparse


def main() -> None:
    """Parse CLI args, build the hyperparameter dict, and run a task."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sd", "dr", "lipo", "qm7"], required=True)
    parser.add_argument(
        "--task", choices=["pretrain", "finetune", "test", "hpopt"], required=True
    )
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--seed", type=float, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    hyperparams = get_hparams(
        args.dataset, args.task, args.experiment_name, args.ckpt, args.lr, args.seed
    )

    pl.seed_everything(hyperparams["seed"], workers=True)
    torch.set_float32_matmul_precision("medium")
    match hyperparams["task"]:
        case "pretrain":
            pretrain(hyperparams)
        case "finetune":
            finetune(hyperparams)
        case "test":
            test(hyperparams)
        case "hpopt":
            optimize_lr(hyperparams)


if __name__ == "__main__":
    main()
