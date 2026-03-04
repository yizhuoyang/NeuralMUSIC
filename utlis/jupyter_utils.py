import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib as mpl

# ------------------------------------------------------------
# Utilities (same as your script)
# ------------------------------------------------------------
def load_P_from_mat(mat_path):
    d = sio.loadmat(str(mat_path))
    for _, v in d.items():
        if isinstance(v, np.ndarray) and v.shape == (3, 4):
            return v.astype(np.float64)

    for _, v in d.items():
        if isinstance(v, np.ndarray):
            vv = np.squeeze(v)
            if vv.shape == (3, 4):
                return vv.astype(np.float64)
            if vv.shape == (4, 3):
                return vv.T.astype(np.float64)

    raise RuntimeError("Cannot find 3x4 P matrix in mat file.")


def project_xyz_to_uv(P, xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim == 1:
        xyz = xyz[None, :]
    Xh = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    x = (P @ Xh.T).T
    uv = x[:, :2] / (x[:, 2:3] + 1e-12)
    return uv


def invert_x_from_deg_and_y(deg, y):
    theta = np.deg2rad(float(deg))
    t = np.tan(theta)
    if abs(t) < 1e-6:
        sign = 1.0 if np.cos(theta) >= 0 else -1.0
        return sign * 1e6
    return float(y) / float(t)


def smooth_prob(prob, win=21):
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    if win is None or win < 3 or (win % 2 == 0):
        return p
    K = p.shape[0]
    pad = win // 2
    p_pad = np.r_[p[-pad:], p, p[:pad]]
    kernel = np.ones(win, dtype=np.float64) / win
    sm = np.convolve(p_pad, kernel, mode="valid")
    return sm[:K]


def ensure_pred_1d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise RuntimeError(f"pred should be (N,) or (N,1), got {arr.shape}")
    return arr.astype(np.float32)


def ensure_prob_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise RuntimeError(f"prob should be (N,K) or (N,1,K), got {arr.shape}")
    return arr.astype(np.float32)


# ------------------------------------------------------------
# Dataset loader (same indexing behavior)
# ------------------------------------------------------------
class AVRootDataset:
    def __init__(self, root, cam=1, image_ext=".png"):
        self.root = Path(root)
        self.cam = cam
        self.image_ext = image_ext
        self.items = []

        for seq in sorted(self.root.iterdir()):
            if not seq.is_dir():
                continue
            gt_dir = seq / "gt"
            img_dir = seq / "image" / f"cam{cam}"
            if not (gt_dir.exists() and img_dir.exists()):
                continue

            for img_path in sorted(img_dir.glob(f"*{self.image_ext}")):
                stem = img_path.stem
                gt_path = gt_dir / f"{stem}.npy"
                if gt_path.exists():
                    self.items.append((img_path, gt_path, seq.name, stem))

        if not self.items:
            raise RuntimeError("No samples indexed. Check ROOT_DIR structure.")
        print(f"[OK] indexed {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def meta(self, idx):
        img_path, gt_path, seq_name, stem = self.items[idx]
        return dict(seq_name=seq_name, stem=stem, image_path=str(img_path), gt_path=str(gt_path))

    def read_image(self, idx):
        img_path = self.items[idx][0]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def read_gt(self, idx):
        gt_path = self.items[idx][1]
        gt = np.load(str(gt_path), allow_pickle=True).item()
        if not isinstance(gt, dict):
            raise RuntimeError(f"GT is not dict: {gt_path}")
        return gt


# ------------------------------------------------------------
# Colors (same)
# ------------------------------------------------------------
RED = (255, 0, 0)
PALETTE = [
    (0, 255, 255),   # cyan
    (255, 255, 0),   # yellow
    (0, 255, 0),     # green
    (0, 128, 255),   # orange-ish
    (255, 255, 255), # white
]


def method_color(method_name, fallback_idx):
    if method_name.lower() == "ours":
        return RED
    return PALETTE[fallback_idx % len(PALETTE)]


# ------------------------------------------------------------
# Drawing helpers (same)
# ------------------------------------------------------------
def get_gt2d_y(gt, person_idx=0):
    gt2d = gt.get("gt2d_xy", None)
    if gt2d is None:
        return None
    gt2d = np.asarray(gt2d)
    if gt2d.ndim == 1:
        return float(gt2d[1])
    if gt2d.ndim == 2 and gt2d.shape[0] > person_idx:
        return float(gt2d[person_idx, 1])
    return None


def apply_bottom_band_background(img_rgb, band_h, alpha=0.2, dark=0.0):
    if alpha <= 0:
        return img_rgb
    out = img_rgb.copy().astype(np.float32)
    H, W = out.shape[:2]
    band_h = int(min(max(1, band_h), H))
    y0 = H - band_h
    base = out[y0:H, :, :]
    bg_val = float(np.clip(dark, 0.0, 1.0)) * 255.0
    bg = np.ones_like(base) * bg_val
    out[y0:H, :, :] = (1 - alpha) * base + alpha * bg
    return out.clip(0, 255).astype(np.uint8)


def draw_dashed_vertical_line(img_rgb, x, y_start, y_end, color_rgb, thickness, dash_len, gap_len):
    out = img_rgb
    H, W = out.shape[:2]
    x = int(round(x))
    if not (0 <= x < W):
        return out

    y0 = int(round(y_start))
    y1 = int(round(y_end))
    y0 = max(0, min(H - 1, y0))
    y1 = max(0, min(H - 1, y1))

    step = int(dash_len) + int(gap_len)
    if step <= 0:
        step = 1

    if y0 <= y1:
        y = y0
        while y <= y1:
            y_dash_end = min(y + int(dash_len) - 1, y1)
            cv2.line(out, (x, y), (x, y_dash_end), color_rgb, int(thickness), cv2.LINE_AA)
            y += step
    else:
        y = y0
        while y >= y1:
            y_dash_end = max(y - int(dash_len) + 1, y1)
            cv2.line(out, (x, y), (x, y_dash_end), color_rgb, int(thickness), cv2.LINE_AA)
            y -= step

    return out


def overlay_prob_shift_peak_to_center_and_align_to_pred_line(
    img_rgb,
    prob_1d,
    u_pred,
    color_rgb,
    band_h,
    smooth_win,
    prob_thickness,
    fill_alpha,
):
    out = img_rgb
    H, W = out.shape[:2]

    p = np.asarray(prob_1d, dtype=np.float64).reshape(-1)
    K = p.shape[0]
    if K < 3:
        return out

    p = np.clip(p, 0.0, None)
    if p.max() > 0:
        p = p / (p.max() + 1e-12)

    p = smooth_prob(p, win=smooth_win)

    peak_bin = int(np.argmax(p))
    center_bin = K // 2
    p = np.roll(p, center_bin - peak_bin)

    anchor_bin = center_bin

    band_h = int(min(max(10, band_h), H))
    y0 = H - band_h
    y_top = y0
    y_bot = H - 1
    peak_y = y_bot - 0.85 * band_h

    px_per_bin = W / float(K)
    bins = np.arange(K, dtype=np.float64)

    xs = float(u_pred) + (bins - float(anchor_bin)) * px_per_bin
    ys = y_bot - p * (y_bot - peak_y)
    ys = np.clip(ys, y_top, y_bot)

    pts = []
    for xv, yv in zip(xs, ys):
        if 0 <= xv < W:
            pts.append([int(round(xv)), int(round(yv))])
    if len(pts) < 2:
        return out

    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(out, [pts], False, color_rgb, int(prob_thickness), cv2.LINE_AA)

    if fill_alpha and fill_alpha > 0:
        overlay = out.copy()
        poly = np.vstack([pts, [pts[-1, 0], y_bot], [pts[0, 0], y_bot]])
        cv2.fillPoly(overlay, [poly.astype(np.int32)], color_rgb)
        out = cv2.addWeighted(out, 1 - float(fill_alpha), overlay, float(fill_alpha), 0)

    return out


def visualize_av16_multi_methods(
    root_dir,
    p_path,
    methods,
    index=0,
    out_path=None,
    cam_id=1,
    person_ref=0,
    band_h=90,
    smooth_win=21,
    line_thickness=4,
    prob_thickness=3,
    ours_line_thickness=6,
    ours_prob_thickness=4,
    dash_len=14,
    gap_len=10,
    pred_stop_below_gt_px=10,
    prob_fill_alpha=0.0,
    band_bg_alpha=0.0,
    band_bg_dark=0.0,
    dpi=180,
    figsize=(10, 6),
    verbose=True,
):
    """
    EXACT same logic as your script, but wrapped as a function.

    Parameters
    ----------
    root_dir : str
        ROOT_DIR in your script.
    p_path : str
        P_PATH in your script (P1.mat).
    methods : dict
        Same METHODS dict:
        {
          "ours": {"pred": "...npy", "prob":"...npy"},
          "doanet": {...},
          ...
        }
    index : int
        sample index.
    out_path : str or None
        if None -> do not save, only return (canvas, meta)
    Returns
    -------
    canvas_rgb : np.ndarray (H,W,3) uint8
    meta : dict
    """

    ds = AVRootDataset(root_dir, cam=cam_id)
    if not (0 <= index < len(ds)):
        raise ValueError(f"index out of range: {index}, len={len(ds)}")

    meta = ds.meta(index)
    img = ds.read_image(index)
    gt = ds.read_gt(index)
    H, W = img.shape[:2]

    # y_gt for ours line stop
    y_gt = get_gt2d_y(gt, person_idx=person_ref)
    if y_gt is None:
        y_gt = H - 1
    y_stop = min(H - 1, int(round(float(y_gt))) + int(pred_stop_below_gt_px))

    # need gt3d to compute u_pred
    xyz = gt.get("gt3d_xyz", None)
    if xyz is None:
        raise RuntimeError("gt3d_xyz missing in GT; cannot compute u_pred.")
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim == 1:
        xyz = xyz[None, :]
    pr = max(0, min(int(person_ref), xyz.shape[0] - 1))
    y_ref = float(xyz[pr, 1])
    z_ref = float(xyz[pr, 2])

    P = load_P_from_mat(p_path)

    # background band once
    canvas = apply_bottom_band_background(img, band_h=band_h, alpha=band_bg_alpha, dark=band_bg_dark)

    # ---- draw order: others first, ours last (ours on top)
    method_items = list(methods.items())
    others = [(n, c) for (n, c) in method_items if n.lower() != "ours"]
    ours_list = [(n, c) for (n, c) in method_items if n.lower() == "ours"]
    ordered = others + ours_list

    for mi, (name, cfg) in enumerate(ordered):
        is_ours = (name.lower() == "ours")
        color = method_color(name, mi)

        # thickness selection
        line_thk = ours_line_thickness if is_ours else line_thickness
        prob_thk = ours_prob_thickness if is_ours else prob_thickness

        pred = ensure_pred_1d(np.load(cfg["pred"], allow_pickle=True))
        pred_deg = float(pred[index]) % 360.0

        # compute u_pred (needed for prob alignment)
        xhat = invert_x_from_deg_and_y(pred_deg, y_ref)
        uv = project_xyz_to_uv(P, np.array([xhat, y_ref, z_ref], dtype=np.float64))
        u_pred = float(uv[0, 0])
        x_pred = int(round(u_pred))

        # ---- ONLY ours draws dashed pred line
        if is_ours:
            canvas = draw_dashed_vertical_line(
                canvas,
                x=x_pred,
                y_start=H - 1,
                y_end=y_stop,
                color_rgb=color,
                thickness=line_thk,
                dash_len=dash_len,
                gap_len=gap_len,
            )

        # ---- prob overlay for all methods (if exists)
        prob_path = cfg.get("prob", None)
        if prob_path is not None and str(prob_path).strip() != "":
            prob = ensure_prob_2d(np.load(prob_path, allow_pickle=True))
            prob_1d = prob[index]  # (K,)

            canvas = overlay_prob_shift_peak_to_center_and_align_to_pred_line(
                canvas,
                prob_1d=prob_1d,
                u_pred=u_pred,
                color_rgb=color,
                band_h=band_h,
                smooth_win=smooth_win,
                prob_thickness=prob_thk,
                fill_alpha=prob_fill_alpha,
            )

    # save (same as your script)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=figsize)
        plt.imshow(canvas)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=int(dpi))
        plt.close()
        if verbose:
            print(f"[OK] saved: {out_path}")

    if verbose:
        print(f"[META] {meta['seq_name']} | {meta['stem']} | idx={index}")

    return canvas, meta


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


def show_canvas(
    canvas,
    title=None,
    meta=None,
    figsize_by_width=10,
    pad=0.02,
    dpi=160,
    add_info=True,
    info_loc="bottom",  # "bottom" or "top"
):
    """
    Nicely display the rendered canvas in Jupyter.
    - keeps aspect ratio
    - removes margins
    - optional title + meta info bar
    """
    H, W = canvas.shape[:2]
    fig_h = figsize_by_width * (H / W)

    fig = plt.figure(figsize=(figsize_by_width, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed
    ax.imshow(canvas)
    ax.set_axis_off()

    if title is not None:
        ax.text(
            0.5, 1.0 - pad if info_loc == "top" else 1.0,
            title,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="none"),
        )

    if add_info and meta is not None:
        info = f"{meta.get('seq_name','')} | {meta.get('stem','')} | idx={meta.get('index','')}"
        y = pad if info_loc == "bottom" else 1.0 - pad
        va = "bottom" if info_loc == "bottom" else "top"
        ax.text(
            0.01, y,
            info,
            transform=ax.transAxes,
            ha="left", va=va,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.35, edgecolor="none"),
            color="white",
        )

    plt.show()
    return fig, ax




def plot_tsne_clusters(
    rx_labels,
    tsne_result,
    figsize=(10, 8),
    dpi=150,
    title=None,
    save_path=None,
    point_size=15,
    alpha=0.8,
    show_center_label=True,
):
    """
    Plot t-SNE clustering result with cluster center annotations.

    Parameters
    ----------
    rx_labels : array-like (N,)
        Cluster labels (e.g., DOA degrees).
    tsne_result : array-like (N, 2)
        2D t-SNE embedding result.
    figsize : tuple
        Figure size.
    dpi : int
        Figure resolution.
    title : str or None
        Optional plot title.
    save_path : str or None
        If provided, save figure to this path.
    point_size : int
        Scatter point size.
    alpha : float
        Scatter transparency.
    show_center_label : bool
        Whether to annotate cluster center with label.
    """

    # ---------------- Style ----------------
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('classic')

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = [
        'DejaVu Sans', 'Helvetica', 'Liberation Sans',
        'Bitstream Vera Sans', 'sans-serif'
    ]
    mpl.rcParams['font.size'] = 30
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    # ---------------- Data ----------------
    rx_labels = np.array(rx_labels).squeeze()
    tsne_result = np.array(tsne_result)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    unique_labels = sorted(np.unique(rx_labels))

    # ---------------- Plot clusters ----------------
    for label in unique_labels:
        idx = (rx_labels == label)
        selected_points = tsne_result[idx]

        if len(selected_points) > 0:
            ax.scatter(
                selected_points[:, 0],
                selected_points[:, 1],
                s=point_size,
                alpha=alpha,
                label=f"{label}°"
            )

            if show_center_label:
                center_x = np.mean(selected_points[:, 0])
                center_y = np.mean(selected_points[:, 1])

                ax.text(
                    center_x, center_y,
                    f"{int(label)}°",
                    fontsize=10,
                    fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.7,
                        edgecolor='none',
                        pad=1.5,
                        boxstyle='round,pad=0.3'
                    )
                )

    # ---------------- Axis styling ----------------
    ax.set_xlabel("t-SNE Dim 1", fontweight='bold')
    ax.set_ylabel("t-SNE Dim 2", fontweight='bold')

    if title is not None:
        ax.set_title(title, fontweight='bold')

    ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()

    return fig, ax

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def plot_tsne_clusters_multi_labels(
    labels_list,
    tsne_results,
    titles=None,
    ncols=2,
    figsize_per_ax=(6, 5),
    dpi=150,
    point_size=15,
    alpha=0.8,
    show_center_label=True,
    share_axes=False,
    show_legend=False,
    legend_loc="best",
    color_mode="global",   # "global" or "per_axes"
    cmap_name="tab20",
    save_path=None,
):
    """
    Plot multiple t-SNE results in subplots, where each subplot has its own labels.

    Parameters
    ----------
    labels_list : list/tuple of array-like
        labels_list[i] is (Ni,) labels for tsne_results[i].
    tsne_results : list/tuple of array-like
        tsne_results[i] is (Ni,2) embedding.
    titles : list[str] or None
        Subplot titles.
    color_mode : str
        "global": same numeric label uses same color across all subplots.
        "per_axes": each subplot uses its own colormap assignment.

    Returns
    -------
    fig, axes
    """

    # ---------------- Style ----------------
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except:
        try:
            plt.style.use("seaborn-whitegrid")
        except:
            plt.style.use("classic")

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "DejaVu Sans", "Helvetica", "Liberation Sans",
        "Bitstream Vera Sans", "sans-serif"
    ]
    mpl.rcParams["font.size"] = 30
    mpl.rcParams["axes.linewidth"] = 1.5
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12

    labels_list = list(labels_list)
    tsne_results = list(tsne_results)
    if len(labels_list) != len(tsne_results):
        raise ValueError(f"labels_list length ({len(labels_list)}) != tsne_results length ({len(tsne_results)})")

    num_plots = len(tsne_results)
    if num_plots == 0:
        raise ValueError("Empty inputs.")

    # ---------------- Global color map (optional) ----------------
    cmap = plt.get_cmap(cmap_name)

    global_color_map = None
    if color_mode == "global":
        all_labels = []
        for labs in labels_list:
            labs = np.array(labs).squeeze()
            all_labels.extend(list(np.unique(labs)))
        global_unique = sorted(set(all_labels))
        global_color_map = {lab: cmap(i % cmap.N) for i, lab in enumerate(global_unique)}
    elif color_mode != "per_axes":
        raise ValueError("color_mode must be 'global' or 'per_axes'.")

    # ---------------- Layout ----------------
    ncols = int(max(1, ncols))
    nrows = int(np.ceil(num_plots / ncols))
    fig_w = figsize_per_ax[0] * ncols
    fig_h = figsize_per_ax[1] * nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        sharex=share_axes,
        sharey=share_axes,
    )

    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel()
    else:
        axes_flat = [axes]

    # ---------------- Plot ----------------
    for i in range(nrows * ncols):
        ax = axes_flat[i]
        if i >= num_plots:
            ax.axis("off")
            continue

        emb = np.asarray(tsne_results[i])
        labs = np.asarray(labels_list[i]).squeeze()

        if emb.ndim != 2 or emb.shape[1] != 2:
            raise ValueError(f"tsne_results[{i}] must be (N,2). Got {emb.shape}.")
        if labs.ndim != 1:
            raise ValueError(f"labels_list[{i}] must be (N,). Got {labs.shape}.")
        if emb.shape[0] != labs.shape[0]:
            raise ValueError(f"Mismatch at {i}: tsne N={emb.shape[0]} vs labels N={labs.shape[0]}.")

        unique_labels = sorted(np.unique(labs))

        # per-axes color assignment if needed
        if color_mode == "per_axes":
            local_color_map = {lab: cmap(j % cmap.N) for j, lab in enumerate(unique_labels)}
        else:
            local_color_map = global_color_map

        for lab in unique_labels:
            idx = (labs == lab)
            pts = emb[idx]
            if len(pts) == 0:
                continue

            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=point_size,
                alpha=alpha,
                color=local_color_map[lab],
                label=f"{lab}°"
            )

            if show_center_label:
                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])
                text_str = f"{int(lab)}°" if np.issubdtype(type(lab), np.number) else str(lab)
                ax.text(
                    cx, cy, text_str,
                    fontsize=10,
                    fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                        pad=1.5,
                        boxstyle="round,pad=0.3"
                    )
                )

        ax.set_xlabel("t-SNE Dim 1", fontweight="bold")
        ax.set_ylabel("t-SNE Dim 2", fontweight="bold")
        if titles is not None and i < len(titles) and titles[i] is not None:
            ax.set_title(titles[i], fontweight="bold")

        ax.grid(True, linestyle="-", alpha=0.3, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        if show_legend:
            ax.legend(loc=legend_loc, frameon=True, fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, axes