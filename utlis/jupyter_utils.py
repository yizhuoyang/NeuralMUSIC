import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_spec_reconstruction(
    dataset,
    model,
    device,
    index=0,
    freq_frac=0.3,
    cmap="viridis"
):
    """
    Visualize GT spec, masked input spec, and reconstructed spec (channel 0).
    
    Parameters
    ----------
    dataset : torch Dataset
    model   : pretrained model
    device  : torch device
    index   : sample index
    freq_frac : float, percentage of frequency range to display (default=0.3)
    cmap    : matplotlib colormap
    """

    # -------------------------
    # Get data
    # -------------------------
    test_sample, gt, mask = dataset[index]
    test_sample = test_sample.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(test_sample)

    # -------------------------
    # Extract channel 0
    # -------------------------
    gt0   = gt[0].detach().cpu().numpy()
    inp0  = test_sample[0, 0].detach().cpu().numpy()
    pred0 = prediction[0, 0].detach().cpu().numpy()

    # -------------------------
    # Crop frequency (first 30%)
    # -------------------------
    F = gt0.shape[0]
    F_keep = max(1, int(F * freq_frac))

    gt0   = gt0[:F_keep, :]
    inp0  = inp0[:F_keep, :]
    pred0 = pred0[:F_keep, :]

    # -------------------------
    # Unified color scale
    # -------------------------
    vmin = min(gt0.min(), inp0.min(), pred0.min())
    vmax = max(gt0.max(), inp0.max(), pred0.max())

    # -------------------------
    # Plot
    # -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    titles = [
        "GT Spectrogram",
        "Masked Input Spectrogram",
        "Reconstructed Spectrogram"
    ]

    specs = [gt0, inp0, pred0]

    for ax, spec, title in zip(axes, specs, titles):

        im = ax.imshow(
            spec.T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("t (Time)")
        ax.set_ylabel("f (Frequency)")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
    cbar.set_label("Magnitude")

    plt.show()