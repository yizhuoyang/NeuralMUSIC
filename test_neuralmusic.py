import argparse
import itertools
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.signal import find_peaks
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_neuralmusic import MIC_PRESETS, build_dataset, build_model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)


def estimate_doa_from_spectrum(spectrum, num_sources, peak_height=0.1, peak_distance=5):
    peaks, _ = find_peaks(spectrum, height=peak_height, distance=peak_distance)
    if len(peaks) >= num_sources:
        selected = peaks[np.argsort(spectrum[peaks])[-num_sources:]]
    else:
        selected = np.argsort(spectrum)[-num_sources:]
    return np.sort(selected % 360)


def circular_mae(pred, target):
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    k = min(len(pred), len(target))
    pred = pred[:k]
    target = target[:k]
    best = float("inf")
    for perm in itertools.permutations(range(k)):
        aligned = pred[list(perm)]
        err = np.abs((aligned - target + 180.0) % 360.0 - 180.0)
        best = min(best, float(err.mean()))
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="Test the proposed NeuralMUSIC model.")
    parser.add_argument("--dataset", choices=["gsc", "soclas", "afpild", "av16"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", help="Dataset root for gsc/soclas/afpild.")
    parser.add_argument("--val-root", help="Validation/test root for av16.")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--mic-preset", choices=sorted(MIC_PRESETS), default="square_4")
    parser.add_argument("--mic-offsets", help="Optional .npy file with shape (num_mics, 3).")
    parser.add_argument("--num-sources", type=int, default=1)
    parser.add_argument("--max-sources", type=int, default=4)
    parser.add_argument("--input-channel", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-percent", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--coherent", type=int, default=1)
    parser.add_argument("--snr", type=int, default=None)
    parser.add_argument("--mode", default="all")
    parser.add_argument("--peak-height", type=float, default=0.1)
    parser.add_argument("--peak-distance", type=int, default=5)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--estimate-num-sources", action="store_true")
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
    args.num_percent = 1.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args, "val")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args, device).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    predictions = []
    ground_truths = []
    spectra = []
    sample_mae = []

    with torch.no_grad():
        for batch_idx, (test_sample, gt, sv, correlation) in enumerate(tqdm(loader, desc="Testing")):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            test_sample = test_sample.to(device).float()
            sv = sv.to(device)
            correlation = correlation.to(device)
            outputs = model(test_sample, sv, correlation)
            batch_spectra = outputs[1].detach().cpu().numpy()

            for spectrum, target in zip(batch_spectra, gt):
                pred = estimate_doa_from_spectrum(
                    spectrum,
                    args.num_sources,
                    peak_height=args.peak_height,
                    peak_distance=args.peak_distance,
                )
                target_np = target.detach().cpu().numpy() % 360
                predictions.append(pred)
                ground_truths.append(target_np)
                spectra.append(spectrum)
                sample_mae.append(circular_mae(pred, target_np))

    mean_mae = float(np.mean(sample_mae)) if sample_mae else float("nan")
    print(f"Circular MAE: {mean_mae:.4f} deg over {len(sample_mae)} samples")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "predictions_neuralmusic.npy", np.array(predictions, dtype=object))
        np.save(save_dir / "ground_truths.npy", np.array(ground_truths, dtype=object))
        np.save(save_dir / "spectra_neuralmusic.npy", np.array(spectra))
        print(f"Saved predictions to {save_dir}")


if __name__ == "__main__":
    main()
