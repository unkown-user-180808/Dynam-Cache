# visualize_utils.py
import numpy as np
import torch
from PIL import Image, ImageDraw
import os
import cv2

def make_patch_overlay(
    image,
    reusable_patch_indices=None,
    critical_patch_indices=None,
    num_patches_per_image: int = 256,
    alpha: float = 0.45,
):
    """Blue: reused/static, yellow: critical, red: recomputed."""
    base = np.asarray(image, dtype=np.uint8)
    if base.ndim != 3 or base.shape[-1] != 3:
        return base

    side = int(np.sqrt(num_patches_per_image))
    if side * side != num_patches_per_image:
        return base

    reusable_patch_indices = set() if reusable_patch_indices is None else set(reusable_patch_indices)
    critical_patch_indices = set() if critical_patch_indices is None else set(critical_patch_indices)
    visual_patch_indices = set(range(num_patches_per_image))
    recompute_patch_indices = visual_patch_indices - reusable_patch_indices - critical_patch_indices

    h, w = base.shape[:2]
    patch_h = max(1, h // side)
    patch_w = max(1, w // side)
    overlay = base.astype(np.float32).copy()

    def paint(indices, color):
        color_arr = np.asarray(color, dtype=np.float32)
        for patch_idx in indices:
            row = int(patch_idx) // side
            col = int(patch_idx) % side
            y0 = row * patch_h
            y1 = h if row == side - 1 else (row + 1) * patch_h
            x0 = col * patch_w
            x1 = w if col == side - 1 else (col + 1) * patch_w
            overlay[y0:y1, x0:x1] = (1.0 - alpha) * overlay[y0:y1, x0:x1] + alpha * color_arr

    paint(recompute_patch_indices, (255, 0, 0))
    paint(reusable_patch_indices, (0, 80, 255))
    paint(critical_patch_indices, (255, 230, 0))

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _to_numpy_rgb(image):
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape={arr.shape}")
    return arr


def attention_map_to_heatmap_overlay(image, attention_map, alpha: float = 0.45):
    base = _to_numpy_rgb(image)

    if attention_map is None:
        return base

    if torch.is_tensor(attention_map):
        attn = attention_map.detach().float().cpu().numpy()
    else:
        attn = np.asarray(attention_map, dtype=np.float32)

    attn = np.squeeze(attn)

    if attn.ndim == 1:
        side = int(np.sqrt(attn.size))
        if side * side != attn.size:
            return base
        attn = attn.reshape(side, side)

    if attn.ndim != 2:
        return base

    attn = np.nan_to_num(attn.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    attn = attn - attn.min()
    attn = attn / (attn.max() + 1e-8)

    h, w = base.shape[:2]
    attn_img = Image.fromarray((attn * 255).astype(np.uint8), mode="L")
    attn_img = attn_img.resize((w, h), resample=Image.BILINEAR)
    attn = np.asarray(attn_img, dtype=np.float32) / 255.0

    heat = np.zeros_like(base, dtype=np.float32)
    heat[..., 0] = 255.0 * attn
    heat[..., 1] = 255.0 * np.clip(1.0 - np.abs(attn - 0.5) * 2.0, 0.0, 1.0)
    heat[..., 2] = 255.0 * (1.0 - attn)

    overlay = (1.0 - alpha) * base.astype(np.float32) + alpha * heat
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _add_title(image, title: str, title_h: int = 24):
    pil = Image.fromarray(_to_numpy_rgb(image))
    canvas = Image.new("RGB", (pil.width, pil.height + title_h), (0, 0, 0))
    canvas.paste(pil, (0, title_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill=(255, 255, 255))

    return canvas


def save_attention_heatmap_grid(
    save_dir: str,
    step: int,
    fixed_image,
    wrist_image,
    fixed_attention_map,
    wrist_attention_map,
    fixed_final_layer_attention_map,
    wrist_final_layer_attention_map,
    fixed_hidden_norm_map=None, wrist_hidden_norm_map=None,
    fixed_patch_overlay=None,  
    wrist_patch_overlay=None,  
    alpha: float = 0.45,
):
    os.makedirs(save_dir, exist_ok=True)

    fixed_attn = attention_map_to_heatmap_overlay(fixed_image, fixed_attention_map, alpha=alpha)
    wrist_attn = attention_map_to_heatmap_overlay(wrist_image, wrist_attention_map, alpha=alpha)
    fixed_final = attention_map_to_heatmap_overlay(fixed_image, fixed_final_layer_attention_map, alpha=alpha)
    wrist_final = attention_map_to_heatmap_overlay(wrist_image, wrist_final_layer_attention_map, alpha=alpha)
    fixed_hidden = attention_map_to_heatmap_overlay(fixed_image, fixed_hidden_norm_map, alpha)
    wrist_hidden = attention_map_to_heatmap_overlay(wrist_image, wrist_hidden_norm_map, alpha)



    

    panels = [
        _add_title(fixed_attn, "Fixed cam - attention"),
        _add_title(wrist_attn, "Wrist cam - attention"),
        _add_title(fixed_final, "Fixed cam - final layer attention"),
        _add_title(wrist_final, "Wrist cam - final layer attention"),
        _add_title(fixed_patch_overlay if fixed_patch_overlay is not None else _to_numpy_rgb(fixed_image), "Fixed cam - patch overlay"),
        _add_title(wrist_patch_overlay if wrist_patch_overlay is not None else _to_numpy_rgb(wrist_image), "Wrist cam - patch overlay"),

    ]

    w = max(panel.width for panel in panels)
    h = max(panel.height for panel in panels)

    resized = [
        panel.resize((w, h), resample=Image.BILINEAR)
        for panel in panels
    ]

    # 3x2 grid
    grid = Image.new("RGB", (w * 2, h * 3), (0, 0, 0))
    for i, panel in enumerate(resized):
        row, col = divmod(i, 2)
        grid.paste(panel, (col * w, row * h))

    path = os.path.join(save_dir, f"step_{step:03d}.png")
    grid.save(path)
    return path

def save_all_layers_heatmap_grid(
    save_dir: str,
    step: int,
    fixed_image,
    wrist_image,
    per_layer_maps,  # dict: {layer_id: (fixed_map, wrist_map)}
    alpha: float = 0.45,
):
    """32개 레이어 전체를 fixed/wrist 쌍으로 저장."""
    os.makedirs(save_dir, exist_ok=True)

    if not per_layer_maps:
        return None

    panels = []
    for layer_id, (fixed_map, wrist_map) in sorted(per_layer_maps.items()):
        panels.append(_add_title(
            attention_map_to_heatmap_overlay(fixed_image, fixed_map, alpha),
            f"Fixed - L{layer_id:02d}"
        ))
        panels.append(_add_title(
            attention_map_to_heatmap_overlay(wrist_image, wrist_map, alpha),
            f"Wrist - L{layer_id:02d}"
        ))

    w = max(p.width  for p in panels)
    h = max(p.height for p in panels)
    resized = [p.resize((w, h), resample=Image.BILINEAR) for p in panels]

    n_rows = len(resized) // 2
    grid = Image.new("RGB", (w * 2, h * n_rows), (0, 0, 0))
    for i, panel in enumerate(resized):
        row, col = divmod(i, 2)
        grid.paste(panel, (col * w, row * h))

    path = os.path.join(save_dir, f"step_{step:03d}_layers.png")
    grid.save(path)
    return path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_entropy_across_steps_plot(
    save_dir: str,
    episode: int,
    entropy_log: dict,
):
    os.makedirs(save_dir, exist_ok=True)
    layer_ids = sorted(entropy_log.keys())
    n_layers = len(layer_ids)
    if n_layers == 0:
        return None

    PHASE_COLORS = {
        'approach':  '#AED6F1',
        'grasp':     '#A9DFBF',
        'lift':      '#F9E79F',
        'transport': '#F0B27A',
        'place':     '#D7BDE2',
        'release':   '#F1948A',
        'unknown':   '#D5D8DC',
    }

    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    axes = axes.flatten()

    for i, layer_id in enumerate(layer_ids):
        ax = axes[i]
        data = entropy_log[layer_id]
        steps  = data['steps']
        phases = data.get('phases', ['unknown'] * len(steps))

        # phase 배경색
        prev_phase = None
        span_start = steps[0]
        for j, (s, p) in enumerate(zip(steps, phases)):
            if p != prev_phase:
                if prev_phase is not None:
                    ax.axvspan(span_start, s,
                               color=PHASE_COLORS.get(prev_phase, '#D5D8DC'),
                               alpha=0.3, zorder=0)
                span_start = s
                prev_phase = p
        if prev_phase is not None:
            ax.axvspan(span_start, steps[-1],
                       color=PHASE_COLORS.get(prev_phase, '#D5D8DC'),
                       alpha=0.3, zorder=0)

        ax.plot(steps, data['fixed'], label='fixed', color='blue', linewidth=1.0, zorder=2)
        ax.plot(steps, data['wrist'], label='wrist', color='orange', linewidth=1.0, zorder=2)
        ax.set_title(f"Layer {layer_id:02d}", fontsize=9)
        ax.set_xlabel("step", fontsize=7)
        ax.set_ylabel("entropy", fontsize=7)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3, zorder=1)

    for j in range(len(layer_ids), len(axes)):
        axes[j].set_visible(False)

    # legend
    from matplotlib.patches import Patch
    used_phases = set()
    for data in entropy_log.values():
        used_phases.update(data.get('phases', []))
    legend_elements = [
        Patch(facecolor=PHASE_COLORS.get(p, '#D5D8DC'), alpha=0.5, label=p)
        for p in PHASE_COLORS if p in used_phases
    ]
    if legend_elements:
        axes[0].legend(handles=legend_elements, loc='upper right', fontsize=6,
                       ncol=len(legend_elements))

    fig.suptitle(f"Episode {episode} - Entropy per Layer across Steps (phase colored)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f"episode_{episode:03d}_entropy_per_layer.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def save_entropy_layers_per_step_grid(
    save_dir: str,
    episode: int,
    snapshots: list,
):
    os.makedirs(save_dir, exist_ok=True)
    if not snapshots:
        return None

    PHASE_COLORS = {
        'approach':  '#AED6F1',
        'grasp':     '#A9DFBF',
        'lift':      '#F9E79F',
        'transport': '#F0B27A',
        'place':     '#D7BDE2',
        'release':   '#F1948A',
        'unknown':   '#D5D8DC',
    }

    n_cols = 4
    n_rows = (len(snapshots) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 5, n_rows * 3),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, snap in zip(axes, snapshots):
        layer_entropies = snap["layer_entropies"]
        layer_ids = sorted(layer_entropies.keys())

        fixed_vals = [layer_entropies[lid]["fixed"] for lid in layer_ids]
        wrist_vals = [layer_entropies[lid]["wrist"] for lid in layer_ids]

        phase = snap.get("phase", "unknown")
        ax.set_facecolor(PHASE_COLORS.get(phase, PHASE_COLORS["unknown"]))

        ax.plot(layer_ids, fixed_vals, marker='o', markersize=2, label='fixed', color='blue', linewidth=1.0)
        ax.plot(layer_ids, wrist_vals, marker='s', markersize=2, label='wrist', color='orange', linewidth=1.0)

        ax.set_title(f"step {snap['step']:03d} | {phase}", fontsize=8)
        ax.set_xlabel("layer", fontsize=7)
        ax.set_ylabel("entropy", fontsize=7)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    for ax in axes[len(snapshots):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"Episode {episode:03d} - Layer Entropy per LLM Step", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(save_dir, f"episode_{episode:03d}_entropy_layers_per_step.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path

def save_eef_trajectory_plot(
    save_dir: str,
    episode: int,
    eef_log: list,
):
    os.makedirs(save_dir, exist_ok=True)
    if not eef_log:
        return None

    steps   = [d['step']     for d in eef_log]
    keys    = ['x', 'y', 'z', 'gripper', 'dx', 'dy', 'dz', 'dgripper']
    labels  = ['x', 'y', 'z', 'gripper', 'Δx', 'Δy', 'Δz', 'Δgripper']

    # phase별 색상
    PHASE_COLORS = {
        'approach':  '#AED6F1',  # 연파랑
        'grasp':     '#A9DFBF',  # 연초록
        'lift':      '#F9E79F',  # 연노랑
        'transport': '#F0B27A',  # 연주황
        'place':     '#D7BDE2',  # 연보라
        'release':   '#F1948A',  # 연빨강
        'unknown':   '#D5D8DC',  # 회색
    }

    fig, axes = plt.subplots(8, 1, figsize=(12, 18), sharex=True)

    for ax, key, label in zip(axes, keys, labels):
        ax.plot(steps, [d[key] for d in eef_log], linewidth=1.2, zorder=2)
        ax.set_ylabel(label, fontsize=8)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', zorder=1)
        ax.grid(True, alpha=0.3, zorder=0)

        # phase 배경색 칠하기
        prev_phase = None
        span_start = steps[0]
        for i, d in enumerate(eef_log):
            curr_phase = d.get('phase', 'unknown')
            if curr_phase != prev_phase:
                if prev_phase is not None:
                    ax.axvspan(span_start, d['step'],
                               color=PHASE_COLORS.get(prev_phase, '#D5D8DC'),
                               alpha=0.3, zorder=0)
                span_start = d['step']
                prev_phase = curr_phase
        # 마지막 구간
        if prev_phase is not None:
            ax.axvspan(span_start, steps[-1],
                       color=PHASE_COLORS.get(prev_phase, '#D5D8DC'),
                       alpha=0.3, zorder=0)

    axes[-1].set_xlabel('step')
    fig.suptitle(f'Episode {episode} - EEF trajectory', fontsize=12)

    # legend 추가
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, alpha=0.5, label=phase)
        for phase, color in PHASE_COLORS.items()
        if any(d.get('phase') == phase for d in eef_log)
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=len(legend_elements))

    plt.tight_layout()
    path = os.path.join(save_dir, f'episode_{episode:03d}_eef_trajectory.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def make_exact_reuse_overlay(
    image,
    exact_reused_patch_indices=None,
    num_patches_per_image: int = 256,
    alpha: float = 0.55,
    pink_color=(255, 105, 180),
):
    """
    modeling_llama.py의 progressive drop에서 '실제로' reuse(=stale, 재계산 안 됨)된
    patch만 핑크색으로 오버레이.

    기존 make_patch_overlay는 run_libero_eval.py가 "이번에 reuse 후보로 넘긴" 것을
    파랑으로 그리는 반면, 이 함수는 modeling_llama.py가 실제로 잘라낸(reuse한)
    patch만 정확히 보여준다.

    Args:
        image: (H, W, 3) uint8 배열, 원본 fixed 또는 wrist 이미지
        exact_reused_patch_indices: 0-based patch index 리스트/set/array
            (이미 fixed_token_start 또는 wrist_token_start가 빼진 상태여야 함)
        num_patches_per_image: 256 (16x16)
        alpha: 핑크 오버레이 강도
        pink_color: 오버레이 색상 (R, G, B)

    Returns:
        np.ndarray (H, W, 3) uint8
    """
    base = np.asarray(image, dtype=np.uint8)
    if base.ndim != 3 or base.shape[-1] != 3:
        return base

    side = int(np.sqrt(num_patches_per_image))
    if side * side != num_patches_per_image:
        return base

    indices = set() if exact_reused_patch_indices is None else set(
        int(i) for i in exact_reused_patch_indices
    )

    h, w = base.shape[:2]
    patch_h = max(1, h // side)
    patch_w = max(1, w // side)
    overlay = base.astype(np.float32).copy()

    color_arr = np.asarray(pink_color, dtype=np.float32)
    for patch_idx in indices:
        if patch_idx < 0 or patch_idx >= num_patches_per_image:
            continue
        row = patch_idx // side
        col = patch_idx % side
        y0 = row * patch_h
        y1 = h if row == side - 1 else (row + 1) * patch_h
        x0 = col * patch_w
        x1 = w if col == side - 1 else (col + 1) * patch_w
        overlay[y0:y1, x0:x1] = (1.0 - alpha) * overlay[y0:y1, x0:x1] + alpha * color_arr

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_exact_reuse_grid(
    save_dir: str,
    step: int,
    fixed_image,
    wrist_image,
    fixed_exact_reused_patch_indices=None,
    wrist_exact_reused_patch_indices=None,
    num_patches_per_image: int = 256,
    alpha: float = 0.55,
):
    """
    fixed/wrist 이미지에 '실제로 reuse된 patch'를 핑크로 오버레이해서
    나란히 붙여 하나의 PNG로 저장.

    Returns:
        저장된 파일 경로 (str)
    """
    os.makedirs(save_dir, exist_ok=True)

    fixed_overlay = make_exact_reuse_overlay(
        fixed_image,
        exact_reused_patch_indices=fixed_exact_reused_patch_indices,
        num_patches_per_image=num_patches_per_image,
        alpha=alpha,
    )
    wrist_overlay = make_exact_reuse_overlay(
        wrist_image,
        exact_reused_patch_indices=wrist_exact_reused_patch_indices,
        num_patches_per_image=num_patches_per_image,
        alpha=alpha,
    )

    n_fixed = len(fixed_exact_reused_patch_indices) if fixed_exact_reused_patch_indices else 0
    n_wrist = len(wrist_exact_reused_patch_indices) if wrist_exact_reused_patch_indices else 0

    panel_fixed = _add_title(fixed_overlay, f"Fixed - exact reuse ({n_fixed}/{num_patches_per_image})")
    panel_wrist = _add_title(wrist_overlay, f"Wrist - exact reuse ({n_wrist}/{num_patches_per_image})")

    w = max(panel_fixed.width, panel_wrist.width)
    h = max(panel_fixed.height, panel_wrist.height)
    panel_fixed = panel_fixed.resize((w, h), resample=Image.BILINEAR)
    panel_wrist = panel_wrist.resize((w, h), resample=Image.BILINEAR)

    grid = Image.new("RGB", (w * 2, h), (0, 0, 0))
    grid.paste(panel_fixed, (0, 0))
    grid.paste(panel_wrist, (w, 0))

    path = os.path.join(save_dir, f"step_{step:03d}_exact_reuse.png")
    grid.save(path)
    return path

def compute_simple_gradient_edge(image, grid_size: int = 16):
    """
    Very lightweight gradient edge map.
    Returns:
        edge_map: [H, W] float32, 0~1
        patch_scores: [grid_size*grid_size] float32
    """
    img = np.asarray(image).astype(np.float32)

    if img.ndim == 3:
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    else:
        gray = img

    gray = gray / 255.0 if gray.max() > 1.0 else gray

    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)

    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    edge_map = np.sqrt(gx ** 2 + gy ** 2)
    edge_map = edge_map / (edge_map.max() + 1e-8)

    h, w = edge_map.shape
    patch_h = max(1, h // grid_size)
    patch_w = max(1, w // grid_size)

    patch_scores = []
    for r in range(grid_size):
        for c in range(grid_size):
            y0 = r * patch_h
            y1 = h if r == grid_size - 1 else (r + 1) * patch_h
            x0 = c * patch_w
            x1 = w if c == grid_size - 1 else (c + 1) * patch_w
            patch_scores.append(float(edge_map[y0:y1, x0:x1].mean()))

    return edge_map, np.asarray(patch_scores, dtype=np.float32)


def save_simple_gradient_edge_grid(
    save_dir: str,
    step: int,
    wrist_image,
    top_ratio: float = 0.20,
):
    """
    Save wrist original + edge debug overlay side by side.
    """
    os.makedirs(save_dir, exist_ok=True)

    img = np.asarray(wrist_image, dtype=np.uint8)
    edge_map, patch_scores = compute_simple_gradient_edge(img)

    threshold = np.quantile(patch_scores, 1.0 - top_ratio)
    edge_patch_indices = np.flatnonzero(patch_scores >= threshold).tolist()

    edge_overlay = make_edge_patch_overlay(
        img,
        edge_patch_indices=edge_patch_indices,
        num_patches_per_image=256,
    )

    panels = [
        _add_title(img, "Wrist original"),
        _add_title(edge_overlay, f"Edge debug top {int(top_ratio * 100)}%"),
    ]

    w = max(p.width for p in panels)
    h = max(p.height for p in panels)

    panels = [p.resize((w, h), resample=Image.BILINEAR) for p in panels]

    grid = Image.new("RGB", (w * 2, h), (0, 0, 0))
    grid.paste(panels[0], (0, 0))
    grid.paste(panels[1], (w, 0))

    path = os.path.join(save_dir, f"step_{step:03d}_simple_gradient_edge.png")
    grid.save(path)

    return path, edge_patch_indices, patch_scores

def compute_coarse_patch_edge(image, grid_size: int = 16):
    """
    Coarse edge from 16x16 patch-mean image.
    This suppresses fine texture edges and keeps object-level boundaries better.
    Returns:
        edge_grid: [grid_size, grid_size] float32, 0~1
        patch_scores: [grid_size*grid_size] float32
    """
    img = np.asarray(image).astype(np.float32)

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape={img.shape}")

    if img.max() > 1.0:
        img = img / 255.0

    h, w = img.shape[:2]
    patch_h = max(1, h // grid_size)
    patch_w = max(1, w // grid_size)

    patch_rgb = np.zeros((grid_size, grid_size, 3), dtype=np.float32)

    for r in range(grid_size):
        for c in range(grid_size):
            y0 = r * patch_h
            y1 = h if r == grid_size - 1 else (r + 1) * patch_h
            x0 = c * patch_w
            x1 = w if c == grid_size - 1 else (c + 1) * patch_w

            patch = img[y0:y1, x0:x1]
            patch_rgb[r, c] = patch.mean(axis=(0, 1))

    gray = (
        0.299 * patch_rgb[..., 0]
        + 0.587 * patch_rgb[..., 1]
        + 0.114 * patch_rgb[..., 2]
    )

    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)

    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    edge_grid = np.sqrt(gx ** 2 + gy ** 2)
    edge_grid = edge_grid / (edge_grid.max() + 1e-8)

    patch_scores = edge_grid.reshape(-1).astype(np.float32)
    return edge_grid, patch_scores


def make_edge_patch_overlay(
    image,
    edge_patch_indices=None,
    num_patches_per_image: int = 256,
    alpha: float = 0.45,
):
    """
    Edge patch만 초록색으로 표시.
    """
    base = np.asarray(image, dtype=np.uint8)
    if base.ndim != 3 or base.shape[-1] != 3:
        return base

    side = int(np.sqrt(num_patches_per_image))
    if side * side != num_patches_per_image:
        return base

    edge_patch_indices = set() if edge_patch_indices is None else set(
        int(i) for i in edge_patch_indices
    )

    h, w = base.shape[:2]
    patch_h = max(1, h // side)
    patch_w = max(1, w // side)

    overlay = base.astype(np.float32).copy()
    color_arr = np.asarray((0, 255, 0), dtype=np.float32)

    for patch_idx in edge_patch_indices:
        if patch_idx < 0 or patch_idx >= num_patches_per_image:
            continue

        row = patch_idx // side
        col = patch_idx % side

        y0 = row * patch_h
        y1 = h if row == side - 1 else (row + 1) * patch_h
        x0 = col * patch_w
        x1 = w if col == side - 1 else (col + 1) * patch_w

        overlay[y0:y1, x0:x1] = (
            (1.0 - alpha) * overlay[y0:y1, x0:x1]
            + alpha * color_arr
        )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_coarse_gradient_edge_grid(
    save_dir: str,
    step: int,
    wrist_image,
    top_ratio: float = 0.10,
    grid_size: int = 16,
):
    """
    Save wrist original + coarse edge patch overlay side by side.
    """
    os.makedirs(save_dir, exist_ok=True)

    img = np.asarray(wrist_image, dtype=np.uint8)
    edge_grid, patch_scores = compute_coarse_patch_edge(
        img,
        grid_size=grid_size,
    )

    threshold = np.quantile(patch_scores, 1.0 - top_ratio)
    edge_patch_indices = np.flatnonzero(patch_scores >= threshold).tolist()

    edge_overlay = make_edge_patch_overlay(
        img,
        edge_patch_indices=edge_patch_indices,
        num_patches_per_image=grid_size * grid_size,
    )

    panels = [
        _add_title(img, "Wrist original"),
        _add_title(edge_overlay, f"Coarse edge top {int(top_ratio * 100)}%"),
    ]

    w = max(p.width for p in panels)
    h = max(p.height for p in panels)

    panels = [p.resize((w, h), resample=Image.BILINEAR) for p in panels]

    grid = Image.new("RGB", (w * 2, h), (0, 0, 0))
    grid.paste(panels[0], (0, 0))
    grid.paste(panels[1], (w, 0))

    path = os.path.join(save_dir, f"step_{step:03d}_coarse_edge.png")
    grid.save(path)

    return path, edge_patch_indices, patch_scores

def make_selected_patch_overlay(
    image,
    patch_indices=None,
    num_patches_per_image: int = 256,
    color=(255, 230, 0),
    alpha: float = 0.45,
):
    """
    선택된 patch만 특정 색으로 표시하는 debug용 overlay.
    make_patch_overlay처럼 나머지 patch를 빨간색으로 칠하지 않는다.
    """
    base = np.asarray(image, dtype=np.uint8)
    if base.ndim != 3 or base.shape[-1] != 3:
        return base

    side = int(np.sqrt(num_patches_per_image))
    if side * side != num_patches_per_image:
        return base

    patch_indices = set() if patch_indices is None else set(
        int(i) for i in patch_indices
    )

    h, w = base.shape[:2]
    patch_h = max(1, h // side)
    patch_w = max(1, w // side)

    overlay = base.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)

    for patch_idx in patch_indices:
        if patch_idx < 0 or patch_idx >= num_patches_per_image:
            continue

        row = patch_idx // side
        col = patch_idx % side

        y0 = row * patch_h
        y1 = h if row == side - 1 else (row + 1) * patch_h
        x0 = col * patch_w
        x1 = w if col == side - 1 else (col + 1) * patch_w

        overlay[y0:y1, x0:x1] = (
            (1.0 - alpha) * overlay[y0:y1, x0:x1]
            + alpha * color_arr
        )

    return np.clip(overlay, 0, 255).astype(np.uint8)

def save_static_boundary_debug_grid(
    save_dir: str,
    step: int,
    llm_call: int,
    wrist_image,
    static_patch_indices=None,
    attention_critical_indices=None,
    boundary_critical_indices=None,
    final_critical_indices=None,
    reuse_patch_indices=None,
    num_patches_per_image: int = 256,
    alpha: float = 0.45,
):
    """
    Static-boundary critical protection 결과를 확인하기 위한 debug grid 저장.

    Panels:
    1. Wrist original
    2. Static patches S
    3. Attention critical C_attn
    4. Boundary critical B = Dilate(P-S) ∩ S
    5. Final critical C_final
    6. Final reuse candidates R = S - C_final
    """
    os.makedirs(save_dir, exist_ok=True)

    img = np.asarray(wrist_image, dtype=np.uint8)

    static_set = set() if static_patch_indices is None else set(
        int(x) for x in static_patch_indices
    )
    attn_set = set() if attention_critical_indices is None else set(
        int(x) for x in attention_critical_indices
    )
    boundary_set = set() if boundary_critical_indices is None else set(
        int(x) for x in boundary_critical_indices
    )

    if final_critical_indices is None:
        final_critical_set = attn_set | boundary_set
    else:
        final_critical_set = set(int(x) for x in final_critical_indices)

    if reuse_patch_indices is None:
        reuse_set = static_set - final_critical_set
    else:
        reuse_set = set(int(x) for x in reuse_patch_indices)

    # 1. original
    original = img

    # 2. static patches
    static_overlay = make_selected_patch_overlay(
        img,
        patch_indices=static_set,
        num_patches_per_image=num_patches_per_image,
        color=(0, 80, 255),  # blue
        alpha=alpha,
    )

    # 3. attention critical
    attn_overlay = make_selected_patch_overlay(
        img,
        patch_indices=attn_set,
        num_patches_per_image=num_patches_per_image,
        color=(255, 230, 0),  # yellow
        alpha=alpha,
    )

    # 4. static-boundary critical
    boundary_overlay = make_selected_patch_overlay(
        img,
        patch_indices=boundary_set,
        num_patches_per_image=num_patches_per_image,
        color=(255, 0, 255),  # magenta
        alpha=alpha,
    )

    # 5. final critical
    final_critical_overlay = make_selected_patch_overlay(
        img,
        patch_indices=static_set,
        num_patches_per_image=num_patches_per_image,
        color=(0, 80, 255),  # blue background: static
        alpha=0.20,
    )
    final_critical_overlay = make_selected_patch_overlay(
        final_critical_overlay,
        patch_indices=final_critical_set,
        num_patches_per_image=num_patches_per_image,
        color=(255, 230, 0),  # yellow: final critical
        alpha=alpha,
    )

    # 6. final reuse candidates
    reuse_overlay = make_selected_patch_overlay(
        img,
        patch_indices=reuse_set,
        num_patches_per_image=num_patches_per_image,
        color=(0, 80, 255),  # blue
        alpha=alpha,
    )

    panels = [
        _add_title(original, "Wrist original"),
        _add_title(static_overlay, f"Static patches S ({len(static_set)})"),
        _add_title(attn_overlay, f"Attention critical ({len(attn_set)})"),
        _add_title(boundary_overlay, f"Boundary critical B ({len(boundary_set)})"),
        _add_title(final_critical_overlay, f"Final critical ({len(final_critical_set)})"),
        _add_title(reuse_overlay, f"Final reuse cand ({len(reuse_set)})"),
    ]

    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    panels = [p.resize((w, h), resample=Image.BILINEAR) for p in panels]

    grid = Image.new("RGB", (w * 3, h * 2), (0, 0, 0))

    for i, panel in enumerate(panels):
        row, col = divmod(i, 3)
        grid.paste(panel, (col * w, row * h))

    path = os.path.join(
        save_dir,
        f"step_{step:03d}_llm_{llm_call:03d}_static_boundary_debug.png",
    )
    grid.save(path)

    stats = {
        "static": len(static_set),
        "attention_critical": len(attn_set),
        "boundary_critical": len(boundary_set),
        "final_critical": len(final_critical_set),
        "reuse": len(reuse_set),
        "path": path,
    }

    return path, stats

def attention_map_to_patch_heatmap_overlay(
    image,
    attention_map,
    alpha: float = 0.45,
    colormap: str = "turbo",
    show_grid_lines: bool = False,
    grid_line_color=(0, 0, 0),
):
    """
    attention_map(16x16 또는 [256] 등)을 patch 단위로 또렷하게 색칠한 heatmap overlay.
    attention_map_to_heatmap_overlay와 달리 보간(bilinear)을 쓰지 않고,
    각 patch가 자신의 점수에 해당하는 단일 색으로 채워진다 (값 분포가 patch별로 명확히 구분됨).

    colormap: "turbo" | "jet" | "viridis" (matplotlib 컬러맵 이름 그대로 사용 가능)
    show_grid_lines: True면 patch 경계에 얇은 선을 그려서 칸 구분을 더 명확하게 함
    """
    base = _to_numpy_rgb(image)

    if attention_map is None:
        return base

    if torch.is_tensor(attention_map):
        attn = attention_map.detach().float().cpu().numpy()
    else:
        attn = np.asarray(attention_map, dtype=np.float32)

    attn = np.squeeze(attn)

    if attn.ndim == 1:
        side = int(np.sqrt(attn.size))
        if side * side != attn.size:
            return base
        attn = attn.reshape(side, side)

    if attn.ndim != 2:
        return base

    side_h, side_w = attn.shape
    attn = np.nan_to_num(attn.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # 0~1 정규화 (값 분포를 그대로 반영)
    attn_min, attn_max = attn.min(), attn.max()
    attn_norm = (attn - attn_min) / (attn_max - attn_min + 1e-8)

    # matplotlib colormap으로 색 매핑 (patch마다 다른 색)
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)
    colored = cmap(attn_norm)[..., :3]  # (side_h, side_w, 3), 0~1 float
    colored = (colored * 255.0).astype(np.uint8)

    h, w = base.shape[:2]
    patch_h = max(1, h // side_h)
    patch_w = max(1, w // side_w)

    overlay = base.astype(np.float32).copy()

    for r in range(side_h):
        for c in range(side_w):
            y0 = r * patch_h
            y1 = h if r == side_h - 1 else (r + 1) * patch_h
            x0 = c * patch_w
            x1 = w if c == side_w - 1 else (c + 1) * patch_w

            color_arr = colored[r, c].astype(np.float32)
            overlay[y0:y1, x0:x1] = (
                (1.0 - alpha) * overlay[y0:y1, x0:x1] + alpha * color_arr
            )

            if show_grid_lines:
                line_color = np.asarray(grid_line_color, dtype=np.float32)
                overlay[y0, x0:x1] = line_color
                overlay[y0:y1, x0] = line_color

    return np.clip(overlay, 0, 255).astype(np.uint8)

def save_text_aware_comparison_grid(
    save_dir: str,
    step: int,
    llm_call: int,
    fixed_image,
    wrist_image,
    fixed_mixed_map,
    wrist_mixed_map,
    fixed_text_only_map,
    wrist_text_only_map,
    alpha: float = 0.45,
    colormap: str = "turbo",        # ← 추가 파라미터
    show_grid_lines: bool = True,   # ← 추가 파라미터, patch 경계를 명확하게
):
    """
    기존(action+text 섞인) attention map과 text-only attention map을
    fixed/wrist 둘 다 나란히 비교하는 2x2 grid 저장.
    patch 단위로 또렷하게 색칠해서 어느 patch가 가장 강한지 한눈에 보이게 함.

    Panels:
    1. Fixed - mixed (action+text)
    2. Fixed - text only
    3. Wrist - mixed (action+text)
    4. Wrist - text only
    """
    os.makedirs(save_dir, exist_ok=True)

    fixed_mixed_overlay = attention_map_to_patch_heatmap_overlay(
        fixed_image, fixed_mixed_map, alpha=alpha, colormap=colormap, show_grid_lines=show_grid_lines,
    )
    fixed_text_overlay = attention_map_to_patch_heatmap_overlay(
        fixed_image, fixed_text_only_map, alpha=alpha, colormap=colormap, show_grid_lines=show_grid_lines,
    )
    wrist_mixed_overlay = attention_map_to_patch_heatmap_overlay(
        wrist_image, wrist_mixed_map, alpha=alpha, colormap=colormap, show_grid_lines=show_grid_lines,
    )
    wrist_text_overlay = attention_map_to_patch_heatmap_overlay(
        wrist_image, wrist_text_only_map, alpha=alpha, colormap=colormap, show_grid_lines=show_grid_lines,
    )

    panels = [
        _add_title(fixed_mixed_overlay, "Fixed - mixed (action+text)"),
        _add_title(fixed_text_overlay, "Fixed - text only"),
        _add_title(wrist_mixed_overlay, "Wrist - mixed (action+text)"),
        _add_title(wrist_text_overlay, "Wrist - text only"),
    ]

    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    panels = [p.resize((w, h), resample=Image.BILINEAR) for p in panels]

    grid = Image.new("RGB", (w * 2, h * 2), (0, 0, 0))
    for i, panel in enumerate(panels):
        row, col = divmod(i, 2)
        grid.paste(panel, (col * w, row * h))

    path = os.path.join(
        save_dir,
        f"step_{step:03d}_llm_{llm_call:03d}_text_aware_compare.png",
    )
    grid.save(path)
    return path

def save_text_aware_three_way_comparison_grid(
    save_dir: str,
    step: int,
    llm_call: int,
    fixed_image,
    wrist_image,
    fixed_mixed_map,              # latest_fixed_spatial_map (action+text 섞인 기존)
    wrist_mixed_map,               # latest_wrist_spatial_map
    fixed_text_only_map,           # latest_fixed_text_only_map (text 전체)
    wrist_text_only_map,           # latest_wrist_text_only_map
    fixed_stopword_filtered_map,   # latest_fixed_stopword_filtered_map (content word만)
    wrist_stopword_filtered_map,   # latest_wrist_stopword_filtered_map
    alpha: float = 0.45,
    colormap: str = "turbo",
    show_grid_lines: bool = True,
):
    """
    mixed(action+text) / text-only / stopword-filtered(content word만)
    세 가지 attention map을 fixed/wrist 둘 다 patch-grid heatmap으로 비교.

    Panels (2행 x 3열):
    1. Fixed - mixed       2. Fixed - text only       3. Fixed - content words only
    4. Wrist - mixed       5. Wrist - text only       6. Wrist - content words only
    """
    os.makedirs(save_dir, exist_ok=True)

    fixed_mixed_overlay = attention_map_to_patch_heatmap_overlay(
        fixed_image, fixed_mixed_map, alpha=alpha, colormap=colormap, show_grid_lines=show_grid_lines,
    )
    fixed_text_overlay = attention_map_to_patch_heatmap_overlay(
        fixed_image, fixed_text_only_map, alpha=alpha, colormap=colormap, show_grid_lines=show_grid_lines,
    )
    fixed_stopword_overlay = attention_map_to_patch_heatmap_overlay(
        fixed_image, fixed_stopword_filtered_map, alpha=alpha, colormap=colormap, show_grid_lines=show_grid_lines,
    )
    wrist_mixed_overlay = attention_map_to_patch_heatmap_overlay(
        wrist_image, wrist_mixed_map, alpha=alpha, colormap=colormap, show_grid_lines=show_grid_lines,
    )
    wrist_text_overlay = attention_map_to_patch_heatmap_overlay(
        wrist_image, wrist_text_only_map, alpha=alpha, colormap=colormap, show_grid_lines=show_grid_lines,
    )
    wrist_stopword_overlay = attention_map_to_patch_heatmap_overlay(
        wrist_image, wrist_stopword_filtered_map, alpha=alpha, colormap=colormap, show_grid_lines=show_grid_lines,
    )

    panels = [
        _add_title(fixed_mixed_overlay, "Fixed - mixed (action+text)"),
        _add_title(fixed_text_overlay, "Fixed - text only"),
        _add_title(fixed_stopword_overlay, "Fixed - content words only"),
        _add_title(wrist_mixed_overlay, "Wrist - mixed (action+text)"),
        _add_title(wrist_text_overlay, "Wrist - text only"),
        _add_title(wrist_stopword_overlay, "Wrist - content words only"),
    ]

    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    panels = [p.resize((w, h), resample=Image.BILINEAR) for p in panels]

    grid = Image.new("RGB", (w * 3, h * 2), (0, 0, 0))
    for i, panel in enumerate(panels):
        row, col = divmod(i, 3)
        grid.paste(panel, (col * w, row * h))

    path = os.path.join(
        save_dir,
        f"step_{step:03d}_llm_{llm_call:03d}_text_aware_3way_compare.png",
    )
    grid.save(path)
    return path

def save_query_mode_attention_grid(
    save_dir: str,
    step: int,
    llm_call: int,
    fixed_image,
    wrist_image,
    fixed_mixed_map,
    wrist_mixed_map,
    fixed_text_only_map,
    wrist_text_only_map,
    fixed_content_words_map,
    wrist_content_words_map,
    fixed_status_only_map,
    wrist_status_only_map,
    fixed_action_only_map,
    wrist_action_only_map,
    alpha: float = 0.45,
    colormap: str = "turbo",
    show_grid_lines: bool = True,
):
    """
    Query mode별 attention map 비교.

    Panels (2행 x 5열):
    Fixed: mixed | text-only | content-only | status-only | action-only
    Wrist: mixed | text-only | content-only | status-only | action-only
    """
    os.makedirs(save_dir, exist_ok=True)

    def overlay_or_blank(image, attn_map):
        if attn_map is None:
            return np.asarray(image, dtype=np.uint8)
        return attention_map_to_patch_heatmap_overlay(
            image,
            attn_map,
            alpha=alpha,
            colormap=colormap,
            show_grid_lines=show_grid_lines,
        )

    fixed_mixed_overlay = overlay_or_blank(fixed_image, fixed_mixed_map)
    fixed_text_overlay = overlay_or_blank(fixed_image, fixed_text_only_map)
    fixed_content_overlay = overlay_or_blank(fixed_image, fixed_content_words_map)
    fixed_status_overlay = overlay_or_blank(fixed_image, fixed_status_only_map)
    fixed_action_overlay = overlay_or_blank(fixed_image, fixed_action_only_map)

    wrist_mixed_overlay = overlay_or_blank(wrist_image, wrist_mixed_map)
    wrist_text_overlay = overlay_or_blank(wrist_image, wrist_text_only_map)
    wrist_content_overlay = overlay_or_blank(wrist_image, wrist_content_words_map)
    wrist_status_overlay = overlay_or_blank(wrist_image, wrist_status_only_map)
    wrist_action_overlay = overlay_or_blank(wrist_image, wrist_action_only_map)

    panels = [
        _add_title(fixed_mixed_overlay, "Fixed - mixed"),
        _add_title(fixed_text_overlay, "Fixed - text only"),
        _add_title(fixed_content_overlay, "Fixed - content words"),
        _add_title(fixed_status_overlay, "Fixed - status only"),
        _add_title(fixed_action_overlay, "Fixed - action only"),

        _add_title(wrist_mixed_overlay, "Wrist - mixed"),
        _add_title(wrist_text_overlay, "Wrist - text only"),
        _add_title(wrist_content_overlay, "Wrist - content words"),
        _add_title(wrist_status_overlay, "Wrist - status only"),
        _add_title(wrist_action_overlay, "Wrist - action only"),
    ]

    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    panels = [p.resize((w, h), resample=Image.BILINEAR) for p in panels]

    grid = Image.new("RGB", (w * 5, h * 2), (0, 0, 0))

    for i, panel in enumerate(panels):
        row = i // 5
        col = i % 5
        grid.paste(panel, (col * w, row * h))

    path = os.path.join(
        save_dir,
        f"step_{step:03d}_llm_{llm_call:03d}_query_modes.png",
    )
    grid.save(path)
    return path

def build_warped_image_from_mapping(keyframe_img, src_x, src_y, img_shape, feat_shape=(16, 16)):
    """
    keyframe_img: (H, W, 3) uint8
    src_x, src_y: (H_f*W_f,) float, keyframe 기준 patch-grid 좌표 (현재 시점 patch가 keyframe의 어디서 왔는지)
    반환: keyframe_img를 현재 시점 기준으로 warp한 (H, W, 3) uint8 이미지
    """
    H_f, W_f = feat_shape
    h, w = img_shape[:2]

    src_x_full = (src_x.reshape(H_f, W_f) / max(W_f - 1, 1)) * (w - 1)
    src_y_full = (src_y.reshape(H_f, W_f) / max(H_f - 1, 1)) * (h - 1)

    map_x = cv2.resize(src_x_full.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(src_y_full.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    warped = cv2.remap(keyframe_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped


def save_attention_warp_2x3_grid(
    save_dir, step, llm_call,
    keyframe_img, current_img, warped_img,
    prev_attention_map, warped_attention_map, current_attention_map,
    colormap="turbo",
):
    def to_heatmap(attn_map, size=224):
        arr = attn_map.detach().cpu().numpy().reshape(16, 16)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        cmap = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_TURBO)
        hm = cv2.applyColorMap((arr * 255).astype(np.uint8), cmap)
        return cv2.resize(hm, (size, size), interpolation=cv2.INTER_NEAREST)

    def add_label(img, label):
        canvas = img.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(canvas, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    rgb_key = cv2.cvtColor(cv2.resize(keyframe_img, (224, 224)), cv2.COLOR_RGB2BGR)
    rgb_warped = cv2.cvtColor(cv2.resize(warped_img, (224, 224)), cv2.COLOR_RGB2BGR)
    rgb_cur = cv2.cvtColor(cv2.resize(current_img, (224, 224)), cv2.COLOR_RGB2BGR)

    hm_prev = to_heatmap(prev_attention_map)
    hm_warped = to_heatmap(warped_attention_map)
    hm_cur = to_heatmap(current_attention_map)

    row1 = np.concatenate([add_label(rgb_key, "Keyframe"), add_label(rgb_warped, "Warped (OF)"), add_label(rgb_cur, "Current")], axis=1)
    row2 = np.concatenate([add_label(hm_prev, "Prev Attn"), add_label(hm_warped, "Warped Attn"), add_label(hm_cur, "Current Attn")], axis=1)
    grid = np.concatenate([row1, row2], axis=0)

    save_path = os.path.join(save_dir, f"step_{step:04d}_call_{llm_call:04d}.png")
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(save_path, grid)
    return save_path

def save_stale_count_heatmap_grid(
    save_dir: str,
    step: int,
    llm_call: int,
    fixed_image,
    wrist_image,
    patch_stale_count: dict,
    fixed_token_start: int,
    wrist_token_start: int,
    num_patches_per_image: int = 256,
    colormap: str = "turbo",
    alpha: float = 0.5,
):
    """
    patch_stale_count(global token index -> 연속 reuse step 수)를
    fixed/wrist 각각 16x16 heatmap으로 시각화.
    값이 높을수록(오래 stale일수록) 진하게 표시되어,
    어느 위치의 patch가 장시간 재계산 안 되고 있는지 한눈에 보여준다.
    """
    os.makedirs(save_dir, exist_ok=True)

    fixed_stale_arr = np.zeros(num_patches_per_image, dtype=np.float32)
    wrist_stale_arr = np.zeros(num_patches_per_image, dtype=np.float32)

    for idx, cnt in patch_stale_count.items():
        if fixed_token_start <= idx < fixed_token_start + num_patches_per_image:
            fixed_stale_arr[idx - fixed_token_start] = cnt
        elif wrist_token_start <= idx < wrist_token_start + num_patches_per_image:
            wrist_stale_arr[idx - wrist_token_start] = cnt

    fixed_stale_map = torch.from_numpy(fixed_stale_arr)
    wrist_stale_map = torch.from_numpy(wrist_stale_arr)

    fixed_overlay = attention_map_to_patch_heatmap_overlay(
        fixed_image, fixed_stale_map, alpha=alpha, colormap=colormap, show_grid_lines=True,
    )
    wrist_overlay = attention_map_to_patch_heatmap_overlay(
        wrist_image, wrist_stale_map, alpha=alpha, colormap=colormap, show_grid_lines=True,
    )

    fixed_max = int(fixed_stale_arr.max())
    wrist_max = int(wrist_stale_arr.max())

    panel_fixed = _add_title(fixed_overlay, f"Fixed - stale (max={fixed_max})")
    panel_wrist = _add_title(wrist_overlay, f"Wrist - stale (max={wrist_max})")

    w = max(panel_fixed.width, panel_wrist.width)
    h = max(panel_fixed.height, panel_wrist.height)
    panel_fixed = panel_fixed.resize((w, h), resample=Image.BILINEAR)
    panel_wrist = panel_wrist.resize((w, h), resample=Image.BILINEAR)

    grid = Image.new("RGB", (w * 2, h), (0, 0, 0))
    grid.paste(panel_fixed, (0, 0))
    grid.paste(panel_wrist, (w, 0))

    path = os.path.join(save_dir, f"step_{step:03d}_llm_{llm_call:03d}_stale_heatmap.png")
    grid.save(path)
    return path

def save_episode_signal_plot(
    save_dir: str,
    episode: int,
    task_description: str,
    step_records: list,
    call_records: list,
    success: bool = None,
):
    '''
    1. EEF x-y trajectory
    2. EEF z + gripper asymmetry
    3. speed / acceleration / jerk
    4. direction cosine + action norm
    5. hole ratio + wrist camera motion
    6. critical / reuse ratio
    7. cache age
    8. actual pruning / exact reuse
    
    '''

    os.makedirs(save_dir, exist_ok=True)
    if not step_records:
        return None

    def get_step(key, default=np.nan):
        vals = []
        for r in step_records:
            v = r.get(key, default)
            vals.append(np.nan if v is None else v)
        return vals

    def get_call(key, default=np.nan):
        vals = []
        for r in call_records:
            v = r.get(key, default)
            vals.append(np.nan if v is None else v)
        return vals

    steps = get_step("step")
    eef_x = get_step("eef_x")
    eef_y = get_step("eef_y")
    eef_z = get_step("eef_z")

    eef_speed = get_step("eef_speed")
    abs_accel = get_step("abs_accel")
    abs_jerk = get_step("abs_jerk")
    direction_cosine = get_step("direction_cosine")

    gripper_width = get_step("gripper_width")
    abs_dgripper_asym = get_step("abs_dgripper_asym")

    action_translation_norm = get_step("action_translation_norm")
    action_rotation_norm = get_step("action_rotation_norm")
    action_eef_ratio = get_step("action_eef_ratio")
    action_eef_ratio_clipped = np.clip(action_eef_ratio, 0, 50)

    recent_net_displacement = get_step("recent_net_displacement")
    recent_path_length = get_step("recent_path_length")
    straightness = get_step("straightness")
    direction_dip_count = get_step("direction_dip_count")
    direction_cosine_std = get_step("direction_cosine_std")

    call_steps = get_call("step")
    hole_ratio = get_call("hole_ratio")
    cam_trans_delta = get_call("cam_trans_delta")
    cam_rot_delta_rad = get_call("cam_rot_delta_rad")

    fixed_crit = get_call("fixed_critical_ratio")
    wrist_crit = get_call("wrist_critical_ratio")
    fixed_reuse = get_call("fixed_reuse_ratio")
    wrist_reuse = get_call("wrist_reuse_ratio")

    fixed_stale_eq_threshold_count = np.asarray(
    get_call("fixed_stale_eq_threshold_count"),
        dtype=np.float32,
    )
    wrist_stale_eq_threshold_count = np.asarray(
        get_call("wrist_stale_eq_threshold_count"),
        dtype=np.float32,
    )
    total_stale_eq_threshold_count = np.asarray(
        get_call("total_stale_eq_threshold_count"),
        dtype=np.float32,
    )

    # old log를 보고 있으면 eq 값이 없어서 전부 NaN일 수 있음
    # 이 경우에는 정확한 eq count는 복구 불가능하므로 0으로 둔다.
    if np.all(np.isnan(fixed_stale_eq_threshold_count)):
        fixed_stale_eq_threshold_count = np.zeros(len(call_steps), dtype=np.float32)

    if np.all(np.isnan(wrist_stale_eq_threshold_count)):
        wrist_stale_eq_threshold_count = np.zeros(len(call_steps), dtype=np.float32)

    if np.all(np.isnan(total_stale_eq_threshold_count)):
        total_stale_eq_threshold_count = (
            fixed_stale_eq_threshold_count + wrist_stale_eq_threshold_count
        )

    fixed_recency_blocked_count = get_call("fixed_recency_blocked_count")
    wrist_recency_blocked_count = get_call("wrist_recency_blocked_count")

    actual_fixed_pruned_ratio = get_call("actual_fixed_pruned_ratio")
    actual_wrist_pruned_ratio = get_call("actual_wrist_pruned_ratio")
    exact_fixed_reuse_ratio = get_call("exact_fixed_reuse_ratio")
    exact_wrist_reuse_ratio = get_call("exact_wrist_reuse_ratio")
    recomputed_candidate_ratio = get_call("recomputed_candidate_ratio")

    fig = plt.figure(figsize=(12, 25))
    gs = fig.add_gridspec(9, 1, height_ratios=[1.3, 1, 1, 1, 1, 1, 1, 1, 1])

    # 1. EEF XY trajectory
    ax = fig.add_subplot(gs[0])
    sc = ax.scatter(eef_x, eef_y, c=steps, s=16, cmap="viridis")
    ax.plot(eef_x, eef_y, linewidth=0.8, alpha=0.6)
    ax.set_title("EEF x-y trajectory")
    ax.set_xlabel("eef_x")
    ax.set_ylabel("eef_y")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="step")

    # 2. EEF z + gripper
    ax = fig.add_subplot(gs[1])
    ax.plot(steps, eef_z, label="eef_z", linewidth=1.0)
    ax.set_ylabel("eef_z")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(steps, abs_dgripper_asym, label="|dgripper_asym|", color="tab:red", linewidth=1.0)
    ax2.plot(steps, gripper_width, label="gripper_width", color="tab:orange", linewidth=1.0, alpha=0.7)
    ax2.set_ylabel("gripper")
    ax.set_title("EEF z + gripper signal")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    # 3. speed / acceleration / jerk
    ax = fig.add_subplot(gs[2])
    ax.plot(steps, eef_speed, label="eef_speed", linewidth=1.0)
    ax.plot(steps, abs_accel, label="|acceleration|", linewidth=1.0)
    ax.plot(steps, abs_jerk, label="|jerk|", linewidth=1.0)
    ax.set_ylabel("motion")
    ax.set_title("Speed, acceleration, jerk")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # 4. progress / oscillation
    ax = fig.add_subplot(gs[3])
    ax.plot(steps, recent_net_displacement, label="recent_net_displacement", linewidth=1.0)
    ax.plot(steps, recent_path_length, label="recent_path_length", linewidth=1.0)
    ax.set_ylabel("distance")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        steps,
        straightness,
        label="straightness",
        color="tab:green",
        linewidth=1.0,
    )
    ax2.plot(
        steps,
        direction_dip_count,
        label="direction_dip_count",
        color="tab:red",
        linewidth=1.0,
        alpha=0.7,
    )
    ax2.plot(
        steps,
        direction_cosine_std,
        label="direction_cosine_std",
        color="tab:orange",
        linewidth=1.0,
        alpha=0.7,
    )
    ax2.set_ylabel("straightness / dip count")

    ax.set_title("Progress / oscillation signal")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    # 5. hole ratio + camera motion
    ax = fig.add_subplot(gs[4])
    ax.plot(steps, direction_cosine, label="direction_cosine", linewidth=1.0)
    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray")
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("direction cosine")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        steps,
        action_translation_norm,
        label="action_translation_norm",
        color="tab:purple",
        linewidth=1.0,
    )
    ax2.plot(
        steps,
        action_rotation_norm,
        label="action_rotation_norm",
        color="tab:brown",
        linewidth=1.0,
        alpha=0.7,
    )
    ax2.plot(
        steps,
        action_eef_ratio_clipped,
        label="action_eef_ratio clipped",
        color="tab:pink",
        linewidth=1.0,
        alpha=0.6,
    )
    ax2.set_ylabel("action norm / ratio")
    ax.set_title("Direction change + action command norm")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    # 6. hole ratio + camera motion
    ax = fig.add_subplot(gs[5])
    if call_records:
        ax.plot(call_steps, hole_ratio, marker="o", label="hole_ratio", linewidth=1.0)
        ax.set_ylabel("hole_ratio")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(
            call_steps,
            cam_trans_delta,
            marker="s",
            linestyle="--",
            label="cam_trans_delta",
            color="tab:green",
            linewidth=1.0,
        )
        ax2.plot(
            call_steps,
            cam_rot_delta_rad,
            marker="^",
            linestyle="--",
            label="cam_rot_delta_rad",
            color="tab:olive",
            linewidth=1.0,
        )
        ax2.set_ylabel("camera motion")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    ax.set_title("Warp hole ratio + wrist camera motion")


    # 7. critical vs reuse
    ax = fig.add_subplot(gs[6])
    if call_records:
        ax.plot(
            call_steps,
            fixed_crit,
            marker="o",
            label="fixed_critical_ratio",
            linewidth=1.0,
        )
        ax.plot(
            call_steps,
            wrist_crit,
            marker="o",
            label="wrist_critical_ratio",
            linewidth=1.0,
        )
        ax.plot(
            call_steps,
            fixed_reuse,
            marker="s",
            linestyle="--",
            label="fixed_reuse_ratio",
            linewidth=1.0,
        )
        ax.plot(
            call_steps,
            wrist_reuse,
            marker="s",
            linestyle="--",
            label="wrist_reuse_ratio",
            linewidth=1.0,
        )

    ax.set_ylabel("%")
    ax.set_title("Critical vs reuse ratio")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


    # 8. stale == threshold index count
    ax = fig.add_subplot(gs[7])
    if call_records:
        ax.plot(
            call_steps,
            fixed_stale_eq_threshold_count,
            marker="o",
            label="fixed indices with stale count == threshold",
            linewidth=1.2,
        )
        ax.plot(
            call_steps,
            wrist_stale_eq_threshold_count,
            marker="o",
            label="wrist indices with stale count == threshold",
            linewidth=1.2,
        )
        ax.plot(
            call_steps,
            total_stale_eq_threshold_count,
            marker="s",
            linestyle="--",
            label="total indices with stale count == threshold",
            linewidth=1.2,
        )

    ax.set_ylabel("# indices")
    ax.set_title("Number of patch indices whose stale count == threshold")
    ax.legend(fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3)

    # 9. actual / exact reuse
    ax = fig.add_subplot(gs[8])
    if call_records:
        ax.plot(
            call_steps,
            exact_fixed_reuse_ratio,
            marker="o",
            label="actual/exact fixed reuse ratio",
            linewidth=1.2,
        )
        ax.plot(
            call_steps,
            exact_wrist_reuse_ratio,
            marker="o",
            label="actual/exact wrist reuse ratio",
            linewidth=1.2,
        )
        ax.plot(
            call_steps,
            recomputed_candidate_ratio,
            marker="^",
            linestyle=":",
            label="recomputed candidate ratio",
            linewidth=1.2,
        )

    ax.set_ylabel("%")
    ax.set_xlabel("step")
    ax.set_title("Actual exact reuse / recomputed candidate ratio")
    ax.legend(fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3)

    status = "SUCCESS" if success else "FAIL"
    color = "#2E7D32" if success else "#C62828"
    fig.suptitle(
        f"Episode {episode} [{status}] - {task_description}",
        fontsize=12,
        color=color,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(save_dir, f"episode_{episode:03d}_signal_summary.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_run_summary_plot(save_dir: str, run_id: str, perf_counters: dict):
    """Run 전체(모든 episode) 기준 fixed/wrist reuse 비율 + layer별 pruning 비율 요약."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    calls = perf_counters.get("llm_reuse_calls", 0)
    if calls > 0:
        fixed_ratio = perf_counters["llm_reusable_fixed_tokens"] / calls / 256 * 100
        wrist_ratio = perf_counters["llm_reusable_wrist_tokens"] / calls / 256 * 100
    else:
        fixed_ratio = wrist_ratio = 0.0
    axes[0].bar(["Fixed", "Wrist"], [fixed_ratio, wrist_ratio], color=["#4C72B0", "#DD8452"])
    for i, v in enumerate([fixed_ratio, wrist_ratio]):
        axes[0].text(i, v + 1, f"{v:.1f}", ha="center")
    axes[0].set_ylabel("Token Reuse Ratio (%)")
    axes[0].set_title("Avg Reuse Ratio by View")

    layer_ids = sorted(perf_counters.get("layer_calls", {}).keys())
    if layer_ids:
        avg_pruned_pct = []
        for lid in layer_ids:
            n = perf_counters["layer_calls"][lid]
            avg_p = perf_counters["layer_pruned_tokens"][lid] / n
            avg_o = perf_counters["layer_original_tokens"][lid] / n
            avg_pruned_pct.append(avg_p / avg_o * 100)
        axes[1].bar([f"L{l}" for l in layer_ids], avg_pruned_pct, color="#55A868")
        axes[1].set_ylabel("Pruned ratio (%)")
        axes[1].set_title("Layer-wise Avg Pruning Ratio")

    fig.suptitle(f"Run summary — {run_id}")
    plt.tight_layout()
    path = os.path.join(save_dir, "run_summary.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

def _adaptive_value_to_text(x, ndigits: int = 6):

    if x is None:
        return "NA"
    try:
        x = float(x)
        if not np.isfinite(x):
            return "NA"
        return f"{x:.{ndigits}f}"
    except Exception:
        return "NA"


def _adaptive_signal_status_line(name, value, threshold):
    import numpy as np

    if value is None:
        return f"{name}: NA | thr={threshold:.6f} -> not counted"

    try:
        v = float(value)
    except Exception:
        return f"{name}: NA | thr={threshold:.6f} -> not counted"

    if not np.isfinite(v):
        return f"{name}: NA | thr={threshold:.6f} -> not counted"

    if v >= threshold:
        return f"{name}: {v:.6f} >= {threshold:.6f} -> RISK +1"
    else:
        return f"{name}: {v:.6f} <  {threshold:.6f} -> ok"


def save_adaptive_phase_debug_frame(
    save_dir: str,
    step: int,
    llm_call: int,
    task_description: str,
    fixed_img,
    wrist_img,
    fixed_overlay_img,
    wrist_overlay_img,
    adaptive_phase_info: dict,
    adaptive_pruning_stats: dict,
    adaptive_thresholds: dict = None,
    target_total_ratios=None,
    adaptive_target_counts=None,
    adaptive_progressive_drop_ratios=None,
    fixed_critical_count: int = 0,
    wrist_critical_count: int = 0,
    fixed_reusable_count: int = 0,
    wrist_reusable_count: int = 0,
    num_patches_per_image: int = 256,
):
    """
    One-image audit for adaptive pruning.
    Left: text explanation.
    Right: fixed/wrist raw + fixed/wrist overlay.
    """
    import os
    import textwrap
    import numpy as np
    from PIL import Image, ImageDraw

    os.makedirs(save_dir, exist_ok=True)

    if adaptive_phase_info is None:
        return None

    phase = adaptive_phase_info.get("global_phase", "NA")
    risk_score = adaptive_phase_info.get("risk_score", "NA")
    reasons = adaptive_phase_info.get("reasons", [])
    fixed_reasons = adaptive_phase_info.get("fixed_reasons", [])
    wrist_reasons = adaptive_phase_info.get("wrist_reasons", [])
    signals = adaptive_phase_info.get("signals", {})

    hole_ratio = signals.get("hole_ratio", None)
    cam_trans_delta = signals.get("cam_trans_delta", None)
    cam_rot_delta_rad = signals.get("cam_rot_delta_rad", None)
    recent_eef_speed = signals.get("recent_eef_speed", None)
    recent_dgripper_asym = signals.get("recent_dgripper_asym", None)

    if phase == "stable_late":
        phase_rule = "risk_score == 0 -> stable_late"
    elif phase == "normal":
        phase_rule = "risk_score == 1 -> normal"
    elif phase == "risky":
        phase_rule = "risk_score >= 2 -> risky"
    else:
        phase_rule = "unknown phase"

    fixed_available = None
    wrist_available = None
    num_candidates = None
    ref_total_tokens = None

    if adaptive_pruning_stats is not None:
        fixed_available = adaptive_pruning_stats.get("fixed_available", None)
        wrist_available = adaptive_pruning_stats.get("wrist_available", None)
        num_candidates = adaptive_pruning_stats.get("num_candidates", None)
        ref_total_tokens = adaptive_pruning_stats.get("ref_total_tokens", None)

    fixed_crit_ratio = fixed_critical_count / num_patches_per_image * 100
    wrist_crit_ratio = wrist_critical_count / num_patches_per_image * 100
    fixed_reuse_ratio = fixed_reusable_count / num_patches_per_image * 100
    wrist_reuse_ratio = wrist_reusable_count / num_patches_per_image * 100

    lines = []
    lines.append(f"Task: {task_description}")
    lines.append(f"step={step} | llm_call={llm_call}")
    lines.append("")
    lines.append(f"PHASE: {phase}")
    lines.append(f"risk_score: {risk_score}")
    lines.append(f"rule: {phase_rule}")
    lines.append(f"reasons: {reasons}")
    lines.append(f"fixed_reasons: {fixed_reasons}")
    lines.append(f"wrist_reasons: {wrist_reasons}")
    lines.append("")
    per_signal_risk = adaptive_phase_info.get("per_signal_risk", None)
    history_lengths = adaptive_phase_info.get("history_lengths", None)
    dominant_signal = adaptive_phase_info.get("dominant_signal", None)

    if per_signal_risk is not None:
        lines.append("[Online normalized risk]")
        lines.append(f"dominant_signal: {dominant_signal}")
        lines.append(f"instant_risk_score: {adaptive_phase_info.get('instant_risk_score', 'NA')}")
        lines.append(f"ema_risk_score: {adaptive_phase_info.get('risk_score', 'NA')}")
        lines.append(f"aggregation: {adaptive_phase_info.get('aggregation', 'NA')}")

        for name in [
            "hole_ratio",
            "cam_trans_delta",
            "cam_rot_delta_rad",
            "recent_eef_speed",
            "recent_dgripper_asym",
        ]:
            value = signals.get(name, None)
            rank = per_signal_risk.get(name, None)
            hlen = history_lengths.get(name, None) if history_lengths is not None else None

            lines.append(
                f"{name}: value={_adaptive_value_to_text(value)} "
                f"| rank={rank:.2f} | hist_len={hlen}"
            )

    elif adaptive_thresholds is not None:
        lines.append("[Signal threshold check]")
        lines.append(_adaptive_signal_status_line(
            "hole_ratio",
            hole_ratio,
            adaptive_thresholds["hole_ratio"],
        ))
        lines.append(_adaptive_signal_status_line(
            "cam_trans_delta",
            cam_trans_delta,
            adaptive_thresholds["cam_trans_delta"],
        ))
        lines.append(_adaptive_signal_status_line(
            "cam_rot_delta_rad",
            cam_rot_delta_rad,
            adaptive_thresholds["cam_rot_delta_rad"],
        ))
        lines.append(_adaptive_signal_status_line(
            "recent_eef_speed",
            recent_eef_speed,
            adaptive_thresholds["recent_eef_speed"],
        ))
        lines.append(_adaptive_signal_status_line(
            "recent_dgripper_asym",
            recent_dgripper_asym,
            adaptive_thresholds["recent_dgripper_asym"],
        ))
    lines.append("")
    lines.append("[Budget]")
    lines.append(f"ref_total_tokens: {ref_total_tokens}")
    lines.append(f"target_total_ratios: {target_total_ratios}")
    lines.append(f"target_counts L2/L6/L10: {adaptive_target_counts}")
    lines.append(f"dynamic_ratios: {adaptive_progressive_drop_ratios}")
    lines.append("")
    lines.append("[Candidates]")
    lines.append(f"fixed_available: {fixed_available}")
    lines.append(f"wrist_available: {wrist_available}")
    lines.append(f"num_candidates: {num_candidates}")
    lines.append("")
    lines.append("[Critical / Reuse]")
    lines.append(f"fixed_critical: {fixed_critical_count}/256 ({fixed_crit_ratio:.1f}%)")
    lines.append(f"wrist_critical: {wrist_critical_count}/256 ({wrist_crit_ratio:.1f}%)")
    lines.append(f"fixed_reuse: {fixed_reusable_count}/256 ({fixed_reuse_ratio:.1f}%)")
    lines.append(f"wrist_reuse: {wrist_reusable_count}/256 ({wrist_reuse_ratio:.1f}%)")

    text = "\n".join(lines)

    def to_pil(img):
        arr = np.asarray(img, dtype=np.uint8)
        return Image.fromarray(arr).convert("RGB").resize((224, 224))

    def add_title(img, title):
        title_h = 24
        canvas = Image.new("RGB", (img.width, img.height + title_h), (0, 0, 0))
        canvas.paste(img, (0, title_h))
        d = ImageDraw.Draw(canvas)
        d.text((5, 5), title, fill=(255, 255, 255))
        return canvas

    fixed_raw = add_title(to_pil(fixed_img), "Fixed raw")
    wrist_raw = add_title(to_pil(wrist_img), "Wrist raw")
    fixed_overlay = add_title(to_pil(fixed_overlay_img), "Fixed overlay")
    wrist_overlay = add_title(to_pil(wrist_overlay_img), "Wrist overlay")

    img_grid_w = 224 * 2
    img_grid_h = fixed_raw.height * 2
    img_grid = Image.new("RGB", (img_grid_w, img_grid_h), (0, 0, 0))
    img_grid.paste(fixed_raw, (0, 0))
    img_grid.paste(wrist_raw, (224, 0))
    img_grid.paste(fixed_overlay, (0, fixed_raw.height))
    img_grid.paste(wrist_overlay, (224, fixed_raw.height))

    text_w = 660
    total_w = text_w + img_grid_w
    total_h = max(img_grid_h, 620)

    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    y = 10
    line_h = 15

    for raw_line in text.split("\n"):
        wrapped = textwrap.wrap(raw_line, width=84) if raw_line else [""]
        for line in wrapped:
            fill = (255, 255, 255)

            if line.startswith("PHASE:"):
                if "risky" in line:
                    fill = (255, 120, 120)
                elif "normal" in line:
                    fill = (255, 220, 120)
                elif "stable_late" in line:
                    fill = (150, 220, 255)

            if "RISK +1" in line:
                fill = (255, 120, 120)
            elif "-> ok" in line:
                fill = (170, 255, 170)

            draw.text((10, y), line, fill=fill)
            y += line_h

    canvas.paste(img_grid, (text_w, 0))

    safe_phase = str(phase).replace("/", "_")
    save_path = os.path.join(
        save_dir,
        f"step_{step:03d}_llm_{llm_call:03d}_{safe_phase}_risk{risk_score}.png",
    )
    canvas.save(save_path)

    return save_path