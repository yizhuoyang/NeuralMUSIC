import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataset.data_loader import AFPILD_raw_Dataset, AV16_Dataset, GSC_Loader, SoClas_database
from model.NeuralMusic import NeuralMusic, NeuralMusic_class
from model.training import NeuralMusicTrainer


MIC_PRESETS = {
    "square_4": np.array(
        [
            [45.7 / 1000 / 2, 45.7 / 1000 / 2, 0.0],
            [-45.7 / 1000 / 2, 45.7 / 1000 / 2, 0.0],
            [-45.7 / 1000 / 2, -45.7 / 1000 / 2, 0.0],
            [45.7 / 1000 / 2, -45.7 / 1000 / 2, 0.0],
        ],
        dtype=np.float32,
    ),
    "av16": np.array(
        [
            [-0.10000, 0.40000, 0.0],
            [-0.07071, 0.32929, 0.0],
            [0.00000, 0.30000, 0.0],
            [0.07071, 0.32929, 0.0],
            [0.10000, 0.40000, 0.0],
            [0.07071, 0.47071, 0.0],
            [0.00000, 0.50000, 0.0],
            [-0.07071, 0.47071, 0.0],
            [-0.10000, -0.40000, 0.0],
            [-0.07071, -0.47071, 0.0],
            [0.00000, -0.50000, 0.0],
            [0.07071, -0.47071, 0.0],
            [0.10000, -0.40000, 0.0],
            [0.07071, -0.32929, 0.0],
            [0.00000, -0.30000, 0.0],
            [-0.07071, -0.32929, 0.0],
        ],
        dtype=np.float32,
    ),
}


def load_mic_offsets(args):
    if args.mic_offsets:
        return np.load(args.mic_offsets).astype(np.float32)
    return MIC_PRESETS[args.mic_preset]


def limit_dataset(dataset, percent):
    if percent >= 1.0:
        return dataset
    if not hasattr(dataset, "items"):
        return dataset
    take = max(1, int(len(dataset.items) * percent))
    dataset.items = dataset.items[:take]
    return dataset


def build_dataset(args, subset):
    mic_offsets = load_mic_offsets(args)
    if args.dataset == "gsc":
        dataset = GSC_Loader(
            root=args.data_root,
            mic_offsets=mic_offsets,
            subset=subset,
            coherent=args.coherent,
            num_source=args.num_sources,
            noise_aug=args.noise_aug and subset == "train",
            geometry_aug=args.geometry_aug,
            num_percent=1.0,
            snr=args.snr,
            mode=args.mode,
        )
        return limit_dataset(dataset, args.num_percent if subset == "train" else args.val_percent)
    if args.dataset == "soclas":
        dataset = SoClas_database(
            root=args.data_root,
            mic_offsets=mic_offsets,
            subset=subset,
            noise_aug=args.noise_aug and subset == "train",
            num_percent=1.0,
        )
        return limit_dataset(dataset, args.num_percent if subset == "train" else args.val_percent)
    if args.dataset == "afpild":
        data_type = "train" if subset == "train" else "test"
        dataset = AFPILD_raw_Dataset(
            dataset_dir=args.data_root,
            mic_offsets=mic_offsets,
            data_type=data_type,
            noise_aug=args.noise_aug and subset == "train",
            num_percent=1.0,
        )
        return limit_dataset(dataset, args.num_percent if subset == "train" else args.val_percent)
    if args.dataset == "av16":
        root = args.train_root if subset == "train" else args.val_root
        dataset = AV16_Dataset(
            processed_root=root,
            mic_offsets=mic_offsets,
            subset=subset,
            noise_aug=args.noise_aug and subset == "train",
            num_percent=1.0,
        )
        return limit_dataset(dataset, args.num_percent if subset == "train" else args.val_percent)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def load_encoder_pretrain(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    encoder_weights = {
        key.replace("encoder.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_weights, strict=False)


def build_model(args, device):
    if args.estimate_num_sources:
        return NeuralMusic_class(
            N=args.max_sources + 1,
            T=1600,
            M=args.num_sources,
            device=device,
            input_channel=args.input_channel,
        )
    return NeuralMusic(
        N=args.max_sources,
        T=1600,
        M=args.num_sources,
        device=device,
        attention=not args.no_attention,
        input_channel=args.input_channel,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the proposed NeuralMUSIC model.")
    parser.add_argument("--dataset", choices=["gsc", "soclas", "afpild", "av16"], required=True)
    parser.add_argument("--data-root", help="Dataset root for gsc/soclas/afpild.")
    parser.add_argument("--train-root", help="Training root for av16.")
    parser.add_argument("--val-root", help="Validation root for av16.")
    parser.add_argument("--save-dir", default="checkpoints/neuralmusic")
    parser.add_argument("--mic-preset", choices=sorted(MIC_PRESETS), default="square_4")
    parser.add_argument("--mic-offsets", help="Optional .npy file with shape (num_mics, 3).")
    parser.add_argument("--num-sources", type=int, default=1)
    parser.add_argument("--max-sources", type=int, default=4)
    parser.add_argument("--input-channel", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-percent", type=float, default=1.0)
    parser.add_argument("--val-percent", type=float, default=1.0)
    parser.add_argument("--coherent", type=int, default=1)
    parser.add_argument("--snr", type=int, default=None)
    parser.add_argument("--mode", default="all")
    parser.add_argument("--noise-aug", action="store_true")
    parser.add_argument("--geometry-aug", action="store_true")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--estimate-num-sources", action="store_true")
    parser.add_argument("--pretrain", help="Optional NeuralMUSIC pretrain checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "av16":
        if not args.train_root or not args.val_root:
            raise ValueError("--train-root and --val-root are required for av16.")
    elif not args.data_root:
        raise ValueError("--data-root is required for gsc/soclas/afpild.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = build_dataset(args, "train")
    val_dataset = build_dataset(args, "val")
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

    model = build_model(args, device)
    if args.pretrain:
        load_encoder_pretrain(model, args.pretrain, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    trainer = NeuralMusicTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        model_dir=args.save_dir,
        device=device,
        lr_scheduler=scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
