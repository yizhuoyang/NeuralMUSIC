import argparse
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.data_loader import (
    AFPILD_raw_Dataset_pretrain,
    AV16_Dataset_pretrain,
    GSP_Loader_pretrain,
    SoClas_database_pretrain,
)
from model.NeuralMusic import NeuralMusic_pretrain
from model.training import NeuralMusicPretrainTrainer


def limit_dataset(dataset, percent):
    if percent >= 1.0:
        return dataset
    if not hasattr(dataset, "items"):
        return dataset
    take = max(1, int(len(dataset.items) * percent))
    dataset.items = dataset.items[:take]
    return dataset


def build_pretrain_dataset(args, subset):
    noise_aug = args.noise_aug and subset == "train"
    if args.dataset == "gsc":
        dataset = GSP_Loader_pretrain(
            root=args.data_root,
            subset=subset,
            coherent=args.coherent,
            num_source=args.num_sources,
            noise_aug=noise_aug,
            geometry_aug=args.geometry_aug,
        )
        return limit_dataset(dataset, args.num_percent if subset == "train" else args.val_percent)
    if args.dataset == "soclas":
        dataset = SoClas_database_pretrain(
            root=args.data_root,
            subset=subset,
            noise_aug=noise_aug,
        )
        return limit_dataset(dataset, args.num_percent if subset == "train" else args.val_percent)
    if args.dataset == "afpild":
        data_type = "train" if subset == "train" else "test"
        dataset = AFPILD_raw_Dataset_pretrain(
            dataset_dir=args.data_root,
            data_type=data_type,
            noise_aug=noise_aug,
        )
        return limit_dataset(dataset, args.num_percent if subset == "train" else args.val_percent)
    if args.dataset == "av16":
        root = args.train_root if subset == "train" else args.val_root
        dataset = AV16_Dataset_pretrain(
            processed_root=root,
            subset=subset,
            noise_aug=noise_aug,
            num_percent=1.0,
        )
        return limit_dataset(dataset, args.num_percent if subset == "train" else args.val_percent)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-supervised NeuralMUSIC masked spectrogram reconstruction pretraining."
    )
    parser.add_argument("--dataset", choices=["gsc", "soclas", "afpild", "av16"], required=True)
    parser.add_argument("--data-root", help="Dataset root for gsc/soclas/afpild.")
    parser.add_argument("--train-root", help="Training root for av16.")
    parser.add_argument("--val-root", help="Validation root for av16.")
    parser.add_argument("--save-dir", default="checkpoints/neuralmusic_pretrain")
    parser.add_argument("--input-channel", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-percent", type=float, default=1.0)
    parser.add_argument("--val-percent", type=float, default=1.0)
    parser.add_argument("--coherent", type=int, default=1)
    parser.add_argument("--num-sources", type=int, default=1)
    parser.add_argument("--noise-aug", action="store_true")
    parser.add_argument("--geometry-aug", action="store_true")
    parser.add_argument("--masked-weight", type=float, default=200.0)
    parser.add_argument("--loss-type", choices=["mse", "l1"], default="mse")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "av16":
        if not args.train_root or not args.val_root:
            raise ValueError("--train-root and --val-root are required for av16.")
    elif not args.data_root:
        raise ValueError("--data-root is required for gsc/soclas/afpild.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = build_pretrain_dataset(args, "train")
    val_dataset = build_pretrain_dataset(args, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = NeuralMusic_pretrain(input_channel=args.input_channel)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)
    trainer = NeuralMusicPretrainTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        model_dir=args.save_dir,
        device=device,
        lr_scheduler=scheduler,
        masked_weight=args.masked_weight,
        loss_type=args.loss_type,
    )
    trainer.train()


if __name__ == "__main__":
    main()
