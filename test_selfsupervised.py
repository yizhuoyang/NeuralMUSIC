import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neuralmusic_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/neuralmusic_cache")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.NeuralMusic import NeuralMusic_pretrain
from model.training import masked_weighted_loss
from train_selfsupervised import build_pretrain_dataset


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)


def save_reconstruction_figure(masked_spec, target_spec, reconstruction, save_path, channel=0, freq_frac=0.3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masked = masked_spec[channel].detach().cpu().numpy()
    target = target_spec[channel].detach().cpu().numpy()
    recon = reconstruction[channel].detach().cpu().numpy()

    f_keep = max(1, int(target.shape[0] * freq_frac))
    masked = masked[:f_keep]
    target = target[:f_keep]
    recon = recon[:f_keep]

    vmin = min(masked.min(), target.min(), recon.min())
    vmax = max(masked.max(), target.max(), recon.max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
    for ax, image, title in zip(
        axes,
        [target, masked, recon],
        ["Target", "Masked input", "Reconstruction"],
    ):
        ax.imshow(image.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Test NeuralMUSIC self-supervised reconstruction.")
    parser.add_argument("--dataset", choices=["gsc", "soclas", "afpild", "av16"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", help="Dataset root for gsc/soclas/afpild.")
    parser.add_argument("--val-root", help="Validation/test root for av16.")
    parser.add_argument("--save-dir", default="results/neuralmusic_pretrain")
    parser.add_argument("--input-channel", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-percent", type=float, default=1.0)
    parser.add_argument("--val-percent", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--coherent", type=int, default=1)
    parser.add_argument("--num-sources", type=int, default=1)
    parser.add_argument("--masked-weight", type=float, default=200.0)
    parser.add_argument("--loss-type", choices=["mse", "l1"], default="mse")
    parser.add_argument("--num-figures", type=int, default=8)
    parser.add_argument("--figure-channel", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "av16":
        if not args.val_root:
            raise ValueError("--val-root is required for av16.")
        args.train_root = args.val_root
    elif not args.data_root:
        raise ValueError("--data-root is required for gsc/soclas/afpild.")

    args.noise_aug = False
    args.geometry_aug = False

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = build_pretrain_dataset(args, "val")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = NeuralMusic_pretrain(input_channel=args.input_channel).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    losses = []
    saved = 0
    with torch.no_grad():
        for batch_idx, (masked_spec, target_spec, mask) in enumerate(tqdm(loader, desc="Reconstruction test")):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            masked_spec = masked_spec.to(device).float()
            target_spec = target_spec.to(device).float()
            mask = mask.to(device).float()
            reconstruction = model(masked_spec)
            loss = masked_weighted_loss(
                reconstruction,
                target_spec,
                mask,
                masked_weight=args.masked_weight,
                loss_type=args.loss_type,
            )
            losses.append(loss.item())

            for i in range(masked_spec.shape[0]):
                if saved >= args.num_figures:
                    break
                save_reconstruction_figure(
                    masked_spec[i],
                    target_spec[i],
                    reconstruction[i],
                    save_dir / f"reconstruction_{saved:03d}.png",
                    channel=args.figure_channel,
                )
                saved += 1

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    np.save(save_dir / "reconstruction_losses.npy", np.array(losses, dtype=np.float32))
    print(f"Mean reconstruction loss: {mean_loss:.6f}")
    print(f"Saved reconstruction outputs to {save_dir}")


if __name__ == "__main__":
    main()
