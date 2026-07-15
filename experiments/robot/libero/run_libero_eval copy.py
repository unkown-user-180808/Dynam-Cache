"""
dynam-cache openvla-oft
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import re
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark
import time
import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    DEVICE,
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from experiments.robot.libero.cache_utils import apply_kv_cache_warp, clone_cache
from experiments.robot.libero.warp_utils import get_sim_handle, get_cam_T_w_c, WarpTracker, compute_static_patch_indices
from experiments.robot.libero.attention_utils import (
    warp_attention_map_with_mapping,
    compute_critical_patch_indices,
    normalize_attention_layer_ids,
    order_candidates_for_pruning,
    apply_view_budget_cap,
    token_attention_merge,
    spatial_scores_to_map,
    get_content_word_row_groups,
    token_attention_merge_word_groups,
    # classify_global_adaptive_pruning_phase,
    # GLOBAL_TOTAL_RATIO_TARGETS,
    # total_ratio_targets_to_candidate_ratios,
    OnlineAdaptiveRiskController,
    build_online_global_adaptive_pruning_plan,
)

from experiments.robot.libero.visualize_utils import (
    make_patch_overlay,
    save_exact_reuse_grid,
    # save_text_aware_comparison_grid,
    # save_text_aware_three_way_comparison_grid,
    # save_query_mode_attention_grid,
    save_stale_count_heatmap_grid,
    save_episode_signal_plot,   
    save_run_summary_plot,      
    save_adaptive_phase_debug_frame,
)



print(f"[LOADED FILE] {__file__}", flush=True)

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================
_CONFIG_LOG_KEYS = (
    "model_family",
    "use_l1_regression",
    "use_diffusion",
    "use_film",
    "use_proprio",
    "num_images_in_input",
    "center_crop",
    "num_open_loop_steps",
    "lora_rank",
    "task_suite_name",
    "num_trials_per_task",
    "seed",
    "local_log_dir",
    "video_save_dir",
    # dynam-cache on/off
    "use_dynam_cache",
    # warp / KV-cache
    "keyframe_interval",
    "use_rolling_anchor",
    "fixed_warp_similarity_threshold",
    "wrist_warp_similarity_threshold",
    "use_cosine_similarity",
    "include_warp_holes",
    "disable_kv_cache_reuse",
    "reuse_mode",
    "use_recency_cap",
    "stale_force_threshold",
    # attention
    "attention_layer_ids",
    "attention_ema_alpha",
    "use_fixed_ema",
    "use_wrist_ema",
    "critical_zscore_k",
    "fixed_critical_zscore_k",
    "wrist_critical_zscore_k",
    # reuse patch selector
    "candidate_ordering_mode",
    "fixed_max_reuse_ratio",  
    "wrist_max_reuse_ratio",   
    # progressive pruning
    "use_adaptive_pruning",
    "progressive_pruning_layers",
    "progressive_drop_ratios",
    "debug_progressive_drop",

    "critical_attention_mode",

    "adaptive_risk_window",
    "adaptive_risk_min_history",
    "adaptive_risk_ema_alpha",
    "adaptive_risk_warmup",
    "adaptive_risk_aggregation",

    "adaptive_min_total_prune_ratios",
    "adaptive_max_total_prune_ratios",
    "adaptive_warmup_fraction",
    "adaptive_exposure_start_cycle",
    "adaptive_exposure_full_cycle",
    "adaptive_motion_weight",
    "adaptive_precision_weight",
    "adaptive_exposure_weight",
)
 
@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    memo: Optional[str] = None                       # Optional memo to include in log file/run id
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    video_save_dir: str = ""

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    
    #  ── Dynam Cache use ──────────────────────────────────────────────────────
    use_dynam_cache: bool = True                                  

     # ── Keyframe / warp ──────────────────────────────────────────────────────
    keyframe_interval: int = 10
    # If True, use previous LLM call as the cache/warp source instead of a fixed keyframe.
    use_rolling_anchor: bool = False
    use_recency_cap: bool = False
    stale_force_threshold: int = 5
    fixed_warp_similarity_threshold: float = 0.90
    wrist_warp_similarity_threshold: float = 0.40
    use_cosine_similarity: bool = True
    include_warp_holes: bool = False
 
    # ── Attention / critical patch ───────────────────────────────────────────
    attention_layer_ids: Tuple[int, ...] = (1,)
    critical_zscore_k: float = 0.20
    # view-specific critical patch threshold 
    fixed_critical_zscore_k: Optional[float] = None # lower k -> more critical patches
    wrist_critical_zscore_k: Optional[float] = None # higher k -> fewer critical patches
 
    # ── Attention EMA ────────────────────────────────────────────────────────
    use_attention_ema: bool = False
    attention_ema_alpha: float = 0.35  # new attention weight: ema = alpha * current + (1-alpha) * previous
    use_fixed_ema: bool = True   # fixed camera에 EMA 적용 여부
    use_wrist_ema: bool = True   # wrist camera에 EMA 적용 여부

    # ── KV cache reuse ───────────────────────────────────────────────────────
    disable_kv_cache_reuse: bool = False
    # ── Reuse mode ───────────────────────────────────────────────────────
    reuse_mode: str = "both"  # "none_prune_off" | "wrist_only" | "fixed_only" | "both"
    
    # Candidate ordering mode for reusable patch pruning:
    # - original: keep default order; fixed tokens come before wrist tokens.
    # - interleave: alternate fixed/wrist candidates to reduce view-order bias.
    # - normattn_global: view-wise normalize attention scores, then globally sort by low attention first.
    # - viewattn_ratioaware: sort each view by attention, then arrange indices to match target pruning ratios.
    candidate_ordering_mode: str = "original"
    # ── View-aware candidate budget cap ──────────────────────────────────────
    fixed_max_reuse_ratio: float = 1.0   # fixed candidate cap (1.0 = no cap)
    wrist_max_reuse_ratio: float = 1.0   # wrist candidate cap (1.0 = no cap)
    
    # ── Grasp-aware / pre-contact critical patch ───────────────────────────
    # use_grasp_aware_critical: bool = False

    # grasp-risk일 때 wrist critical 주변만 확장
    # grasp_wrist_dilation_radius: int = 1

    # # 로그 기반 초기값
    # grasp_eef_speed_thr: float = 0.005
    # grasp_gripper_close_thr: float = 0.0005
    # grasp_gripper_open_thr: float = 0.070

    # ── Progressive layer drop ───────────────────────────────────────────────
    use_adaptive_pruning: bool = False
    progressive_pruning_layers: Optional[Tuple[int, ...]] = None
    progressive_drop_ratios: Optional[Tuple[float, ...]] = None
    debug_progressive_drop: bool = False

    # ── Text-aware critical patch (text query만 사용해서 attention map 생성) ──
    # use_text_aware_critical: bool = False
    # # ── Text-aware debug visualization ───────────────────────────────────────
    # use_text_aware_debug: bool = False
    # # ── Action→text attention 1차 검증 (로그 전용, critical patch에 영향 없음) ──
    # use_action_to_text_debug: bool = False
    # # ── Stopword 제외 text attention map 검증 (로그+캐시 전용, critical patch에 영향 없음) ──
    # use_stopword_filtered_debug: bool = False
    # # ── Query-mode attention map 비교
    # # mixed / text-only / content-only / status-only / action-only를 각각 visualize
    # use_query_mode_map_debug: bool = False
    # ── Critical patch에 사용할 attention source ─────────────────────
    # "mixed"         : 기존 action+text 전체 query map
    # "text_only"     : prompt text 전체 query map
    # "content_words" : task content words only map
    # "status_only"   : robot proprio/status token map
    # "action_only"   : action tokens map
    critical_attention_mode: str = "mixed"

    # ── Online adaptive risk controller ─────────────────────────────────────
    adaptive_risk_window: int = 20
    adaptive_risk_min_history: int = 8
    adaptive_risk_ema_alpha: float = 0.35
    adaptive_risk_warmup: float = 0.5
    adaptive_risk_aggregation: str = "top2_mean"  # "max" | "mean" | "top2_mean"

    # ── Adaptive pruning envelope ─────────────────────────────────────
    # min: long task에서 성공률을 지킨 conservative floor
    # max: goal/spatial/object에서 성공률을 지킨 efficiency ceiling
    adaptive_min_total_prune_ratios: Tuple[float, ...] = (0.078, 0.235, 0.313)
    adaptive_max_total_prune_ratios: Tuple[float, ...] = (0.158, 0.290, 0.485)

    # ── Recency-cap-derived exposure schedule ─────────────────────────
    # warmup/exposure는 absolute call 수가 아니라 stale_force_threshold 기준 cycle로 정의
    adaptive_warmup_fraction: float = 1.0
    adaptive_exposure_start_cycle: float = 1.0
    adaptive_exposure_full_cycle: float = 2.5

    # ── Risk composition ──────────────────────────────────────────────
    # motion은 main risk signal에서 제외
    adaptive_motion_weight: float = 0.0
    adaptive_precision_weight: float = 1.0
    adaptive_exposure_weight: float = 1.0


    # fmt: on

# =============================================================================
# Logging helpers
# =============================================================================
 
def _sanitize_for_filename(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return sanitized.strip("._-")
 
 
def _write_run_header(log_file, cfg: "GenerateConfig", run_id: str) -> None:
    log_file.write(f"Run ID: {run_id}\n")
    if cfg.memo is not None and cfg.memo.strip():
        log_file.write(f"Memo: {cfg.memo.strip()}\n")
    log_file.write("Config:\n")
    for key in _CONFIG_LOG_KEYS:
        log_file.write(f"  {key}: {getattr(cfg, key)}\n")
    log_file.write("\n")
    log_file.flush()
 
 
def _log_llama_perf_metrics(log_file, model, perf_counters = None) -> None:
    if log_file is None:
        return

    language_model = getattr(model, "language_model", None)
    llama_model = getattr(language_model, "model", None)
    if llama_model is None:
        return

    current_cuda_latency = getattr(llama_model, "last_cuda_time", None)
    average_cuda_latency = getattr(llama_model, "last_cuda_time_avg", None)
    average_tflops = getattr(llama_model, "last_tflops_avg", None)

    if None not in (current_cuda_latency, average_cuda_latency, average_tflops):
        log_file.write(
            f"Current CUDA latency: {float(current_cuda_latency):.6f} ms | "
            f"Average CUDA latency: {float(average_cuda_latency):.6f} ms, "
            f"Average TFLOPs: {float(average_tflops):.6f}\n"
        )
        log_file.flush()

    q_ratios = getattr(llama_model, "last_q_ratios", None)
    if q_ratios:
        layer_lines = []
        for k, v in sorted((kk, vv) for kk, vv in q_ratios.items() if kk != 'total'):
            layer_lines.append(
                f"L{k}: remaining={v['remaining']:.1f}% pruned={v['pruned']:.1f}% "
                f"({v['pruned_tokens']}/{v['seq_length']})"
            )
        if layer_lines:
            log_file.write(f"Q Ratio (per layer): {' | '.join(layer_lines)}\n")
        if 'total' in q_ratios:
            tot = q_ratios['total']
            fixed_pruned = tot.get('fixed_pruned', 0)
            wrist_pruned = tot.get('wrist_pruned', 0)
            total_visual_pruned = fixed_pruned + wrist_pruned
            fixed_ratio = fixed_pruned / 256 * 100
            wrist_ratio = wrist_pruned / 256 * 100
            total_visual_ratio = total_visual_pruned / 512 * 100
            log_file.write(
                f"Q Ratio (total): remaining={tot['remaining']:.1f}% pruned={tot['pruned']:.1f}% "
                f"({tot['pruned_tokens']}/{tot['original_seq_length']}) | "
                f"Fixed pruned={fixed_ratio:.1f}% ({fixed_pruned}/256) | "
                f"Wrist pruned={wrist_ratio:.1f}% ({wrist_pruned}/256) | "
                f"Visual pruned={total_visual_ratio:.1f}% ({total_visual_pruned}/512)\n"
            )

        if perf_counters is not None:
            for k, v in q_ratios.items():
                if k == 'total':
                    continue
                perf_counters["layer_calls"][k] = perf_counters["layer_calls"].get(k, 0) + 1
                perf_counters["layer_pruned_tokens"][k] = perf_counters["layer_pruned_tokens"].get(k, 0) + v["pruned_tokens"]
                perf_counters["layer_original_tokens"][k] = perf_counters["layer_original_tokens"].get(k, 0) + v["original_seq_length"]
                perf_counters["layer_fixed_pruned"][k] = perf_counters["layer_fixed_pruned"].get(k, 0) + v.get("fixed_pruned", 0)
                perf_counters["layer_wrist_pruned"][k] = perf_counters["layer_wrist_pruned"].get(k, 0) + v.get("wrist_pruned", 0)

            if 'total' in q_ratios:
                tot = q_ratios['total']
                perf_counters["total_pruning_calls"] += 1
                perf_counters["total_pruned_tokens_sum"] += tot["pruned_tokens"]
                perf_counters["total_original_tokens_sum"] += tot["original_seq_length"]
                perf_counters["total_fixed_pruned_sum"] += tot.get("fixed_pruned", 0)
                perf_counters["total_wrist_pruned_sum"] += tot.get("wrist_pruned", 0)
        log_file.flush()
 
 
def log_message(message: str, log_file=None) -> None:
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def _count_reusable_tokens_by_view(
    token_indices,
    fixed_token_start: int,
    wrist_token_start: int,
    num_patches_per_image: int,
) -> Tuple[int, int]:
    if not token_indices:
        return 0, 0

    fixed_token_end = fixed_token_start + num_patches_per_image
    wrist_token_end = wrist_token_start + num_patches_per_image

    fixed_count = sum(fixed_token_start <= int(idx) < fixed_token_end for idx in token_indices)
    wrist_count = sum(wrist_token_start <= int(idx) < wrist_token_end for idx in token_indices)
    return fixed_count, wrist_count


def _token_indices_to_patch_indices(token_indices, token_start: int, num_patches: int = 256):
    if not token_indices:
        return set()
    token_end = token_start + num_patches
    return {
        int(idx) - token_start
        for idx in token_indices
        if token_start <= int(idx) < token_end
    }

def _safe_direction_cosine(curr_delta, prev_delta, eps: float = 1e-8):
    if curr_delta is None or prev_delta is None:
        return np.nan

    curr = np.asarray(curr_delta, dtype=np.float32)
    prev = np.asarray(prev_delta, dtype=np.float32)

    curr_norm = float(np.linalg.norm(curr))
    prev_norm = float(np.linalg.norm(prev))

    if curr_norm < eps or prev_norm < eps:
        return np.nan

    return float(np.dot(curr, prev) / (curr_norm * prev_norm + eps))


def _rotation_delta_angle_rad(T_prev, T_curr):
    """
    Camera rotation change angle in radians.
    T_prev, T_curr: 4x4 camera pose matrix.
    """
    if T_prev is None or T_curr is None:
        return np.nan

    R_prev = np.asarray(T_prev[:3, :3], dtype=np.float32)
    R_curr = np.asarray(T_curr[:3, :3], dtype=np.float32)

    R_delta = R_curr @ R_prev.T
    cos_theta = (np.trace(R_delta) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


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

    fixed_max_age = int(fixed_stale_arr.max())
    wrist_max_age = int(wrist_stale_arr.max())

    threshold = 8  # cfg.stale_force_threshold랑 맞추고 싶으면 함수 인자로 넘기는 게 더 좋음

    fixed_eq_count = int((fixed_stale_arr == threshold).sum())
    wrist_eq_count = int((wrist_stale_arr == threshold).sum())

    fixed_ge_count = int((fixed_stale_arr >= threshold).sum())
    wrist_ge_count = int((wrist_stale_arr >= threshold).sum())

    panel_fixed = _add_title(
        fixed_overlay,
        f"Fixed - stale eq{threshold}={fixed_eq_count}, ge{threshold}={fixed_ge_count}, max_age={fixed_max_age}",
    )
    panel_wrist = _add_title(
        wrist_overlay,
        f"Wrist - stale eq{threshold}={wrist_eq_count}, ge{threshold}={wrist_ge_count}, max_age={wrist_max_age}",
    )

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

    call_steps = get_call("step")
    hole_ratio = get_call("hole_ratio")
    cam_trans_delta = get_call("cam_trans_delta")
    cam_rot_delta_rad = get_call("cam_rot_delta_rad")

    fixed_crit = get_call("fixed_critical_ratio")
    wrist_crit = get_call("wrist_critical_ratio")
    fixed_reuse = get_call("fixed_reuse_ratio")
    wrist_reuse = get_call("wrist_reuse_ratio")

    fixed_cache_age_mean = get_call("fixed_cache_age_mean")
    wrist_cache_age_mean = get_call("wrist_cache_age_mean")
    fixed_cache_age_max = get_call("fixed_cache_age_max")
    wrist_cache_age_max = get_call("wrist_cache_age_max")

    actual_fixed_pruned_ratio = get_call("actual_fixed_pruned_ratio")
    actual_wrist_pruned_ratio = get_call("actual_wrist_pruned_ratio")
    exact_fixed_reuse_ratio = get_call("exact_fixed_reuse_ratio")
    exact_wrist_reuse_ratio = get_call("exact_wrist_reuse_ratio")
    recomputed_candidate_ratio = get_call("recomputed_candidate_ratio")

    fig = plt.figure(figsize=(12, 22))
    gs = fig.add_gridspec(8, 1, height_ratios=[1.3, 1, 1, 1, 1, 1, 1, 1])

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

    # 4. direction + action
    ax = fig.add_subplot(gs[3])
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

    # 5. hole ratio + camera motion
    ax = fig.add_subplot(gs[4])
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

    # 6. critical vs reuse
    ax = fig.add_subplot(gs[5])
    if call_records:
        ax.plot(call_steps, fixed_crit, marker="o", label="fixed_critical_ratio")
        ax.plot(call_steps, wrist_crit, marker="o", label="wrist_critical_ratio")
        ax.plot(call_steps, fixed_reuse, marker="s", linestyle="--", label="fixed_reuse_ratio")
        ax.plot(call_steps, wrist_reuse, marker="s", linestyle="--", label="wrist_reuse_ratio")
    ax.set_ylabel("%")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Critical vs reuse ratio")
    ax.grid(True, alpha=0.3)

    # 7. cache age
    ax = fig.add_subplot(gs[6])
    if call_records:
        ax.plot(call_steps, fixed_cache_age_mean, marker="o", label="fixed_age_mean")
        ax.plot(call_steps, wrist_cache_age_mean, marker="o", label="wrist_age_mean")
        ax.plot(call_steps, fixed_cache_age_max, marker="s", linestyle="--", label="fixed_age_max")
        ax.plot(call_steps, wrist_cache_age_max, marker="s", linestyle="--", label="wrist_age_max")
    ax.set_ylabel("stale count")
    ax.set_title("Cache age / stale duration")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 8. actual pruning / exact reuse
    ax = fig.add_subplot(gs[7])
    if call_records:
        ax.plot(call_steps, actual_fixed_pruned_ratio, marker="o", label="actual_fixed_pruned_ratio")
        ax.plot(call_steps, actual_wrist_pruned_ratio, marker="o", label="actual_wrist_pruned_ratio")
        ax.plot(call_steps, exact_fixed_reuse_ratio, marker="s", linestyle="--", label="exact_fixed_reuse_ratio")
        ax.plot(call_steps, exact_wrist_reuse_ratio, marker="s", linestyle="--", label="exact_wrist_reuse_ratio")
        ax.plot(call_steps, recomputed_candidate_ratio, marker="^", linestyle=":", label="recomputed_candidate_ratio")
    ax.set_ylabel("%")
    ax.set_xlabel("step")
    ax.set_title("Actual pruning / exact reuse / recomputed candidate ratio")
    ax.legend(fontsize=7, ncol=2)
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

def _cache_age_stats(
    patch_stale_count,
    token_start: int,
    num_patches: int = 256,
    stale_threshold: int = 8,
):
    ages = []
    token_end = token_start + num_patches

    for idx in range(token_start, token_end):
        ages.append(float(patch_stale_count.get(idx, 0)))

    ages = np.asarray(ages, dtype=np.float32)

    eq_cap_count = int((ages == stale_threshold).sum())
    ge_cap_count = int((ages >= stale_threshold).sum())

    return {
        "mean": float(ages.mean()),
        "max_age": float(ages.max()),

        # 네가 원하는 값
        "eq_cap_count": eq_cap_count,

        # recency cap 걸릴 후보 수 볼 때 유용한 값
        "ge_cap_count": ge_cap_count,

        "eq_cap_ratio": eq_cap_count / num_patches * 100,
        "ge_cap_ratio": ge_cap_count / num_patches * 100,
    }

# =============================================================================
# Validation / initialization
# =============================================================================

def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)
 
    # attention layer ids 정규화
    cfg.attention_layer_ids = normalize_attention_layer_ids(cfg.attention_layer_ids)
 
    if cfg.model_family == "openvla":
        if cfg.use_dynam_cache:
            # progressive pruning 설정 주입
            model.language_model.config.progressive_pruning_layers = (
                list(cfg.progressive_pruning_layers) if cfg.progressive_pruning_layers else None
            )
            model.language_model.config.progressive_drop_ratios = (
                [float(x) for x in cfg.progressive_drop_ratios] if cfg.progressive_drop_ratios else None
            )
            model.language_model.config.debug_progressive_drop = cfg.debug_progressive_drop
    
            # attention 수집 layer 설정
            if cfg.disable_kv_cache_reuse:
                model.language_model.config.collect_attn_layers = None   # 전체 layer
            else:
                model.language_model.config.collect_attn_layers = list(cfg.attention_layer_ids)
        else:
            model.language_model.config.progressive_pruning_layers = None
            model.language_model.config.progressive_drop_ratios = None
            model.language_model.config.debug_progressive_drop = False
            model.language_model.config.collect_attn_layers = None
            model.language_model.config.reusable_patches = None
            model.language_model.config.warped_past_key_values = None
            model.language_model.config.force_cache_output = False
            model.language_model.config.force_attention_output = False
    
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
 
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)
 
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)
 
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)
 
    return model, action_head, proprio_projector, noisy_action_projector, processor

def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key

# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    if cfg.memo is not None and cfg.memo.strip():
        memo_tag = _sanitize_for_filename(cfg.memo)
        if memo_tag:
            run_id += f"--memo_{memo_tag}"
 
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # ── video 저장 폴더 설정 ──────────────────────────────────────────
    if not cfg.video_save_dir:
        cfg.video_save_dir = os.path.join(cfg.local_log_dir, run_id)
    os.makedirs(cfg.video_save_dir, exist_ok=True)
    logger.info(f"Saving rollout videos to: {cfg.video_save_dir}")
    # ─────────────────────────────────────────────────────────────────
 
    _write_run_header(log_file, cfg, run_id)
 
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
 
    return log_file, local_log_filepath, run_id

# =============================================================================
# Observation / action helpers
# =============================================================================

def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    
    log_message("Using default initial states", log_file)
    return initial_states, None

# obs 얻을 준비.
def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action

# =============================================================================
# Warping Error Check Functions
# =============================================================================

def rotate_half_debug(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_debug(x, cos, sin):
    return (x * cos) + (rotate_half_debug(x) * sin)


def remove_rope_debug(x, cos, sin):
    return (x * cos) - (rotate_half_debug(x) * sin)


# =============================================================================
# Episode runner
# =============================================================================

def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    total_episodes: int = 0,
    perf_counters: Optional[dict] = None,
):
    """
    Run a single episode in the environment.
    warp / KV-cache reuse / attention critical-patch 로직 포함.
    action chunk (open-loop) 방식은 oft 고유 방식 유지.
    """
    if perf_counters is None:
        perf_counters = {"total_steps": 0, "total_time": 0.0}
    perf_counters.setdefault("total_wall_ms", 0.0)
    perf_counters.setdefault("total_all_steps", 0)
    perf_counters.setdefault("total_reusable_fixed_tokens", 0)
    perf_counters.setdefault("total_reusable_wrist_tokens", 0)

    perf_counters.setdefault("llm_reuse_calls", 0)
    perf_counters.setdefault("llm_reusable_fixed_tokens", 0)
    perf_counters.setdefault("llm_reusable_wrist_tokens", 0)
    perf_counters.setdefault("real_control_total_time_s", 0.0)
    perf_counters.setdefault("real_control_total_actions", 0)


    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue | action chunk 크기 불일치 경고
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        logger.warning(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    # attention_heatmap_dir = os.path.join(
    #     cfg.video_save_dir,
    #     "attention_map_heatmap_overlay",
    #     f"episode_{total_episodes + 1:03d}",
    # )
    # # ── VISUALIZE REUSE PATCH ─────────────────────────

    exact_reuse_dir = os.path.join(
        cfg.video_save_dir,
        "exact_reuse_patch",
        f"episode_{total_episodes + 1:03d}",
    )
    # # ── VISUALIZE REUSE PATCH ─────────────────────────

    # # ── VISUALIZE stale count heatmap ─────────────────────────
    stale_heatmap_dir = os.path.join(
        cfg.video_save_dir,
        "stale_heatmap",
        f"episode_{total_episodes + 1:03d}",
    )
    # #── adaptive image ─────────────────────────
    adaptive_phase_debug_dir = os.path.join(
        cfg.video_save_dir,
        "adaptive_phase_debug",
        f"episode_{total_episodes + 1:03d}",
    )
    # #── adaptive image end─────────────────────────
    os.makedirs(adaptive_phase_debug_dir, exist_ok=True)
    # # ── VISUALIZE stale count heatmap ─────────────────────────

    replay_images = []
    replay_images_wrist = []
    replay_images_heatmap = []
    replay_images_wrist_heatmap = []
    last_caches = None
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]


    # -------------------- [PHASE 구간] --------------------------------
    # eef_log = []  # {'step': t, 'x': , 'y': , 'z': , 'gripper': }
    # prev_eef_obs = None  # 이전 step obs 저장용
    # gripper_history = []
    # gripper_baseline = None
    # last_phase = "unkown"

    # ── Signal logging setup ────────────────────────────────
    prev_eef_obs = None
    recent_eef_speeds = deque(maxlen=cfg.num_open_loop_steps)
    recent_dgripper_asyms = deque(maxlen=cfg.num_open_loop_steps)
    recent_abs_accels = deque(maxlen=cfg.num_open_loop_steps)
    recent_abs_jerks = deque(maxlen=cfg.num_open_loop_steps)
    recent_speed_drop_ratios = deque(maxlen=cfg.num_open_loop_steps)
    recent_action_eef_ratios = deque(maxlen=cfg.num_open_loop_steps)
    step_signal_records = []   # ← 추가
    call_signal_records = []   # ← 추가
    prev_delta_pos = None
    prev_step_eef_speed = None
    prev_step_accel = None

    # LLM-call 단위 wrist camera motion 계산용
    prev_signal_T_w_c = None

    # Progress / oscillation window
    progress_window = max(4, cfg.num_open_loop_steps)

    recent_eef_positions = deque(maxlen=progress_window + 1)
    recent_direction_cosines = deque(maxlen=progress_window)
    # ──────────────────────────────────────────────────────────
    adaptive_risk_controller = None

    if cfg.use_adaptive_pruning:
        adaptive_risk_controller = OnlineAdaptiveRiskController(
            window_size=cfg.adaptive_risk_window,
            min_history=cfg.adaptive_risk_min_history,
            ema_alpha=cfg.adaptive_risk_ema_alpha,
            warmup_risk=cfg.adaptive_risk_warmup,
            aggregation=cfg.adaptive_risk_aggregation,

            warmup_fraction=cfg.adaptive_warmup_fraction,
            exposure_start_cycle=cfg.adaptive_exposure_start_cycle,
            exposure_full_cycle=cfg.adaptive_exposure_full_cycle,

            motion_weight=cfg.adaptive_motion_weight,
            precision_weight=cfg.adaptive_precision_weight,
            exposure_weight=cfg.adaptive_exposure_weight,
        )

    # Run episode
    success = False

    # -------------------------------------------------------------------------
    # Episode-local warp / KV-cache state
    # -------------------------------------------------------------------------
    if cfg.use_dynam_cache:
        warp_tracker     = WarpTracker(keyframe_interval=cfg.keyframe_interval)
    else:
        warp_tracker     = None
    fixed_sim_thr        = cfg.fixed_warp_similarity_threshold
    wrist_sim_thr        = cfg.wrist_warp_similarity_threshold
    last_caches          = {}       # get_action이 반환하는 per-step 캐시 (attention map 등)
    keyframe_cache       = None     # 현재 keyframe의 KV cache
    warped_caches        = None     # keyframe KV를 현재 frame으로 warp한 것
    keyframe_T_w_c       = None     # keyframe 카메라 pose
    keyframe_img         = None     # keyframe wrist camera 이미지

    # Rolling-anchor state: previous LLM call cache/image/pose.
    # When cfg.use_rolling_anchor=True, these three are used together as the KV source
    # and geometric warp source, so cache source and warp source stay consistent.
    cache_anchor_cache   = None
    cache_anchor_T_w_c   = None
    cache_anchor_img     = None

    prev_attn_T_w_c      = None     # 직전 LLM 호출 때의 wrist camera pose
    prev_attn_wrist_img  = None     # 직전 LLM 호출 때의 wrist image
    prev_llm_fixed_img   = None     # 직전 LLM 호출 때의 fixed image
    llm_call_count    = 0           # action chunk / LLM 호출 횟수
    prev_fixed_attention_key = (
        "ema_fixed_spatial_map"
        if getattr(cfg, "use_attention_ema", False) and getattr(cfg, "use_fixed_ema", False)
        else "latest_fixed_spatial_map"
    )

    prev_wrist_attention_key = (
        "ema_wrist_spatial_map"
        if getattr(cfg, "use_attention_ema", False) and getattr(cfg, "use_wrist_ema", False)
        else "latest_wrist_spatial_map"
    )
    last_heatmap_fixed = None   # 마지막 LLM 호출 시의 fixed heatmap
    last_heatmap_wrist = None   # 마지막 LLM 호출 시의 wrist heatmap

    num_patches_per_image = 256
    fixed_token_start = 1
    wrist_token_start = fixed_token_start + num_patches_per_image

    # ── stale duration 추적용 ─────────────────────────────────
    # key: token index (fixed/wrist 글로벌 인덱스), value: 연속 reuse(=재계산 안 됨) step 수
    patch_stale_count = {}
    # ────────────────────────────────────────────────────────
    # # [clone_cache_with_raw]
    # rotary_emb_module = None
    # if cfg.use_dynam_cache:
    #     rotary_emb_module = model.language_model.model.layers[0].self_attn.rotary_emb

    all_visual_token_indices = list(
        range(fixed_token_start, fixed_token_start + num_patches_per_image)
    ) + list(
        range(wrist_token_start, wrist_token_start + num_patches_per_image)
    )
    episode_total_wall_ms = 0.0
    episode_all_steps = 0

    real_control_total_time_s = 0.0
    real_control_total_actions = 0

    current_chunk_inference_s = 0.0
    current_chunk_env_s = 0.0
    current_chunk_actions_executed = 0
    current_chunk_size = 0
    chunk_wall_start = None 

    
    try:
        while t < max_steps + cfg.num_steps_wait:
            # -----------------------------------------------------------------
            # [Step 0] 초기 대기 (물체 안정화)
            # -----------------------------------------------------------------

            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            
            step_start_time = time.perf_counter()
 
            # -----------------------------------------------------------------
            # [Step 1] 이미지 / pose 획득
            # -----------------------------------------------------------------
            observation, fixed_img = prepare_observation(obs, resize_size)
            wrist_img = observation["wrist_image"]
            _t_step1 = time.perf_counter()

            is_keyframe_step = False
            T_w_c_curr = None
            final_reusable_indices = None
            applied_reusable_indices = None
            task_relevant_indices = None
            adaptive_progressive_drop_ratios = None
            adaptive_pruning_stats = None
            adaptive_phase_info = None
            adaptive_target_counts = None
            target_total_ratios = None
            ref_total_tokens = None

            adaptive_cam_trans_delta = None
            adaptive_cam_rot_delta_rad = None
            adaptive_recent_eef_speed = None
            adaptive_recent_dgripper_asym = None

            # adaptive_signal_thresholds = {
            #     "hole_ratio": 0.15,
            #     "cam_trans_delta": 0.015,
            #     "cam_rot_delta_rad": 0.08,
            #     "recent_eef_speed": 0.010,
            #     "recent_dgripper_asym": 0.010,
            # }
            fixed_task_relevant_indices = []
            wrist_task_relevant_indices = []
            fixed_static_indices = None
            wrist_static_indices = None
            static_indices = None
            hole_ratio = None

            kv_warp_mapping = None
            # kv_within_bounds = None
            # attn_within_bounds = None

            # recovered_hole_indices = []
            # critical_hole_indices = []
            # unknown_hole_indices = []

            wrist_task_relevant_patch_indices = []
            fixed_task_relevant_patch_indices = []
            wrist_attention_critical_patch_indices = set()

            reusable_count = 0
            fixed_reusable_count = 0
            wrist_reusable_count = 0
            force_keyframe_refresh = False
            hard_refresh_step = False
            will_call_llm = len(action_queue) == 0

            fixed_recency_blocked_count = 0
            wrist_recency_blocked_count = 0
            fixed_recency_blocked_ratio = 0.0
            wrist_recency_blocked_ratio = 0.0


            # # grasp-aware
            # grasp_risk = False
            # grasp_approach = False
            # grasp_transition = False
            # grasp_phase = current_grasp_phase
            # curr_gripper_width_for_grasp = 0.0
            # recent_max_eef_speed = 0.0
            # recent_min_dgripper = 0.0


            if cfg.use_dynam_cache and will_call_llm:

                next_llm_call_count = llm_call_count + 1
                force_keyframe_refresh = (
                    cfg.keyframe_interval > 0
                    and next_llm_call_count % cfg.keyframe_interval == 0
                )

                is_keyframe_step = (t == cfg.num_steps_wait)
                if cfg.use_rolling_anchor:
                    # Rolling-anchor mode has no periodic keyframe full-refresh.
                    # First LLM call initializes the rolling anchor; afterwards recency cap handles patch refresh.
                    hard_refresh_step = is_keyframe_step
                else:
                    hard_refresh_step = (
                        is_keyframe_step
                        or force_keyframe_refresh  # and not cfg.use_recency_cap
                    )
                sim        = get_sim_handle(env)
                warp_tracker.init_env_info(sim, wrist_img.shape)
                T_w_c_curr = get_cam_T_w_c(sim, warp_tracker.cam_id)

                # 최초 step: anchor 초기화
                if keyframe_T_w_c is None:
                    keyframe_T_w_c, keyframe_img = T_w_c_curr, wrist_img

                if cfg.use_rolling_anchor and cache_anchor_T_w_c is None:
                    cache_anchor_T_w_c, cache_anchor_img = T_w_c_curr, wrist_img
                
                if prev_attn_T_w_c is None:
                    prev_attn_T_w_c, prev_attn_wrist_img = T_w_c_curr, wrist_img
    
                # -----------------------------------------------------------------
                # [Step 2] Warping
                # -----------------------------------------------------------------
                final_reusable_indices   = None
                task_relevant_indices    = None
                fixed_task_relevant_indices = []
                wrist_task_relevant_indices = []
                static_indices           = None
                warped_ema_attention_map = None
    
                if hard_refresh_step:
                    # 진짜 full recompute step
                    warped_caches = None
                    final_reusable_indices = None
                    applied_reusable_indices = None
                    model.language_model.config.reusable_patches = None

                    log_message(
                        f"[HARD REFRESH] step={t:03d} "
                        f"llm_call={next_llm_call_count} "
                        f"is_keyframe_step={is_keyframe_step} "
                        f"force_keyframe_refresh={force_keyframe_refresh} "
                        f"use_recency_cap={cfg.use_recency_cap} "
                        f"T_w_c_curr_none={T_w_c_curr is None}",
                        log_file,
                    )
                else:
                    # ── image warp (keyframe → current, optical flow는 prev→current incremental) ──
                    
                    if cfg.use_rolling_anchor:
                        # Rolling-anchor KV warp:
                        # previous LLM frame cache/image/pose -> current LLM frame
                        if cache_anchor_T_w_c is None or cache_anchor_img is None:
                            kv_warp_mapping, wrist_static_indices = None, None
                        else:
                            kv_warp_mapping, wrist_static_indices = warp_tracker.step_warp(
                                cache_anchor_T_w_c,
                                cache_anchor_img,
                                T_w_c_curr,
                                wrist_img,
                                img_shape=wrist_img.shape[:2],
                                threshold=wrist_sim_thr,
                                use_cosine_similarity=cfg.use_cosine_similarity,
                                include_warp_holes=cfg.include_warp_holes,
                            )
                    else:
                        # Original keyframe KV warp:
                        # keyframe frame -> current frame
                        kv_warp_mapping, wrist_static_indices = warp_tracker.step_warp(
                            keyframe_T_w_c, keyframe_img, T_w_c_curr, wrist_img,
                            img_shape=wrist_img.shape[:2],
                            threshold=wrist_sim_thr,
                            use_cosine_similarity=cfg.use_cosine_similarity,
                            include_warp_holes=cfg.include_warp_holes,
                        )

                    # ── hole ratio 계산 ──────────────────────────
                    if kv_warp_mapping is not None and warp_tracker.last_within_bounds is not None:
                        hole_ratio = 1.0 - warp_tracker.last_within_bounds.float().mean().item()
                    else:
                        hole_ratio = None
                    # ──────────────────────────────────────────────────────────

                    # get fixed camera static_indicies
                    fixed_static_indices = None
                    if prev_llm_fixed_img is not None:
                        fixed_static_indices = compute_static_patch_indices(
                            prev_llm_fixed_img, 
                            fixed_img, 
                            threshold=fixed_sim_thr,
                            use_cosine_similarity=cfg.use_cosine_similarity,
                        )


                    # ── attention map warp (prev_llm → current) ───────────────────────
                    if (
                        last_caches.get(prev_wrist_attention_key) is not None
                        and prev_attn_T_w_c is not None
                        and prev_attn_wrist_img is not None
                    ):
                        attn_warp_mapping, attn_within_bounds = warp_tracker.step_warp_geom_only(
                            prev_attn_T_w_c,
                            prev_attn_wrist_img,
                            T_w_c_curr,
                            img_shape=wrist_img.shape[:2],
                        )

                        warped_ema_attention_map = warp_attention_map_with_mapping(
                            prev_attn_map=last_caches[prev_wrist_attention_key],
                            warp_mapping=attn_warp_mapping,
                            within_bounds=attn_within_bounds,
                        )

                        # print visualization
                        if warped_ema_attention_map is not None:
                            last_caches["latest_warped_wrist_attn_map"] = warped_ema_attention_map.detach().cpu()
                    else:
                        warped_ema_attention_map = None

                # -----------------------------------------------------------------
                # [Step 3] Reuse patch 선정
                # -----------------------------------------------------------------
                # if warped_ema_attention_map is not None:
                #     wrist_task_relevant_patch_indices = compute_critical_patch_indices(
                #         warped_ema_attention_map,
                #         zscore_k=cfg.critical_zscore_k,
                #     )
                #     wrist_task_relevant_indices = [
                #         wrist_token_start + idx for idx in wrist_task_relevant_patch_indices
                #     ]

                #     if last_caches.get(prev_fixed_attention_key) is not None:
                #         fixed_task_relevant_patch_indices = compute_critical_patch_indices(
                #             last_caches[prev_fixed_attention_key],
                #             zscore_k=cfg.critical_zscore_k,
                #         )
                #         fixed_task_relevant_indices = [
                #             fixed_token_start + idx for idx in fixed_task_relevant_patch_indices
                #         ]

                #     task_relevant_indices = list(set(wrist_task_relevant_indices) | set(fixed_task_relevant_indices))
                
                wrist_task_relevant_indices = []
                fixed_task_relevant_indices = []

                # view-specific critical zscore threshold
                fixed_critical_zscore_k = (
                    cfg.fixed_critical_zscore_k
                    if cfg.fixed_critical_zscore_k is not None
                    else cfg.critical_zscore_k
                )

                wrist_critical_zscore_k = (
                    cfg.wrist_critical_zscore_k
                    if cfg.wrist_critical_zscore_k is not None
                    else cfg.critical_zscore_k
                )

                # # -----------------------------------------------------------------
                # # [Step 2.5] Pre-contact / grasp-transition detection
                # # -----------------------------------------------------------------
                # if cfg.use_grasp_aware_critical:
                #     curr_gripper_qpos = obs["robot0_gripper_qpos"]
                #     curr_gripper_width_for_grasp = float(curr_gripper_qpos[0] - curr_gripper_qpos[1])

                #     recent_max_eef_speed = (
                #         float(max(recent_eef_speeds)) if len(recent_eef_speeds) > 0 else 0.0
                #     )
                #     recent_min_dgripper = (
                #         float(min(recent_dgrippers)) if len(recent_dgrippers) > 0 else 0.0
                #     )

                #     is_gripper_open = curr_gripper_width_for_grasp > cfg.grasp_gripper_open_thr
                #     is_moving = recent_max_eef_speed > cfg.grasp_eef_speed_thr
                #     is_closing = recent_min_dgripper < -cfg.grasp_gripper_close_thr

                #     grasp_approach = bool(is_gripper_open and is_moving)
                #     grasp_transition = bool(is_gripper_open and is_closing)
                #     grasp_risk = bool(grasp_approach or grasp_transition)

                #     if grasp_transition:
                #         grasp_phase = "grasp_transition"
                #     elif grasp_approach:
                #         grasp_phase = "pre_contact"
                #     elif (not is_gripper_open) and is_moving:
                #         grasp_phase = "closed_transport"
                #     else:
                #         grasp_phase = "normal"

                #     # 이번 LLM call / action chunk에 적용될 phase 저장
                #     current_grasp_phase = grasp_phase
                #     current_grasp_risk = grasp_risk

                # -----------------------------------------------------------------
                # Wrist critical patch
                # -----------------------------------------------------------------
                wrist_task_relevant_patch_indices = []

                if warped_ema_attention_map is not None:
                    
                    wrist_task_relevant_patch_indices = compute_critical_patch_indices(
                        warped_ema_attention_map,
                        zscore_k=wrist_critical_zscore_k,
                    )

                    # attention/edge critical만 따로 저장
                    wrist_attention_critical_patch_indices = set(
                        int(x) for x in wrist_task_relevant_patch_indices
                    )

                    # # Grasp-aware dilation
                    # if (
                    #     cfg.use_grasp_aware_critical
                    #     and grasp_risk
                    #     and cfg.grasp_wrist_dilation_radius > 0
                    # ):
                    #     wrist_task_relevant_patch_indices = dilate_patch_indices(
                    #         wrist_task_relevant_patch_indices,
                    #         radius=cfg.grasp_wrist_dilation_radius,
                    #         grid_size=16,
                    #     )


                # 여기서 딱 한 번만 token index로 변환
                wrist_task_relevant_indices = [
                    wrist_token_start + idx for idx in wrist_task_relevant_patch_indices
                ]

                # fixed critical patch
                if last_caches.get(prev_fixed_attention_key) is not None:
                    fixed_task_relevant_patch_indices = compute_critical_patch_indices(
                        last_caches[prev_fixed_attention_key],
                        zscore_k=fixed_critical_zscore_k,
                    )
                    fixed_task_relevant_indices = [
                        fixed_token_start + idx for idx in fixed_task_relevant_patch_indices
                    ]

                # # ── GRASP PHASE DEBUG ─────────────────────────────────────
                # if cfg.use_grasp_aware_critical and next_llm_call_count <= 50:
                #     log_message(
                #         f"[GRASP DBG] step={t:03d} llm_call={next_llm_call_count} "
                #         f"phase={grasp_phase} "
                #         f"risk={grasp_risk} approach={grasp_approach} transition={grasp_transition} "
                #         f"gripper_width={curr_gripper_width_for_grasp:.5f} "
                #         f"recent_max_eef_speed={recent_max_eef_speed:.5f} "
                #         f"recent_min_dgripper={recent_min_dgripper:.5f} "
                #         f"wrist_dilation_radius={cfg.grasp_wrist_dilation_radius if grasp_risk else 0} "
                #         f"fixed_critical={len(fixed_task_relevant_indices)} "
                #         f"wrist_critical={len(wrist_task_relevant_indices)}",
                #         log_file,
                #     )
                # # ─────────────────────────────────────────────────────────


                if wrist_task_relevant_indices or fixed_task_relevant_indices:
                    task_relevant_indices = list(
                        set(wrist_task_relevant_indices) | set(fixed_task_relevant_indices)
                    )
                else:
                    task_relevant_indices = None

                # -----------------------------------------------------------------
                # Final static token indices
                # fixed static + wrist static, including recovered low-risk holes
                # -----------------------------------------------------------------
                static_indices = []

                if fixed_static_indices is not None:
                    static_indices.extend([fixed_token_start + idx for idx in fixed_static_indices])

                if wrist_static_indices is not None:
                    static_indices.extend([wrist_token_start + idx for idx in wrist_static_indices])

                if len(static_indices) == 0:
                    static_indices = None

                # -----------------------------------------------------------------
                # KV cache warp after optional attention-based hole recovery
                # -----------------------------------------------------------------
                source_cache = cache_anchor_cache if cfg.use_rolling_anchor else keyframe_cache

                if cfg.disable_kv_cache_reuse or source_cache is None:
                    warped_caches = None
                elif cfg.reuse_mode in ("fixed_only", "none_prune_off"):
                    warped_caches = clone_cache(source_cache)
                else:
                    rotary_emb_module = model.language_model.model.layers[0].self_attn.rotary_emb

                    if kv_warp_mapping is None:
                        warped_caches = None
                    else:
                        warp_mapping_t = torch.as_tensor(
                            kv_warp_mapping,
                            device=DEVICE,
                            dtype=torch.long,
                        )

                        warped_caches = apply_kv_cache_warp(
                            past_key_values_input=source_cache,
                            warp_mapping=warp_mapping_t,
                            rotary_emb_module=rotary_emb_module,
                            v_token_start=wrist_token_start,
                        )
                
                recency_blocked = set()
                if static_indices is not None:
                    if task_relevant_indices is not None:
                        candidates = list(set(static_indices) - set(task_relevant_indices))
                    else:
                        candidates = list(static_indices)


                   # ── recency cap: N번 연속 reuse된 patch는 이번 call에서 제외 ──
                    if cfg.use_recency_cap and patch_stale_count:
                        recency_blocked_all = {
                            idx for idx, cnt in patch_stale_count.items()
                            if cnt >= cfg.stale_force_threshold
                        }

                        # 실제 이번 call의 reuse 후보 중에서 recency cap 때문에 빠지는 patch만 count
                        recency_blocked_candidates = set(candidates) & recency_blocked_all

                        fixed_recency_blocked_count, wrist_recency_blocked_count = _count_reusable_tokens_by_view(
                            list(recency_blocked_candidates),
                            fixed_token_start,
                            wrist_token_start,
                            num_patches_per_image,
                        )

                        fixed_recency_blocked_ratio = fixed_recency_blocked_count / num_patches_per_image * 100
                        wrist_recency_blocked_ratio = wrist_recency_blocked_count / num_patches_per_image * 100

                        if recency_blocked_candidates:
                            candidates = [idx for idx in candidates if idx not in recency_blocked_all]

                            log_message(
                                f"[RECENCY CAP] step={t:03d} llm_call={next_llm_call_count} "
                                f"blocked_total_candidates={len(recency_blocked_candidates)} "
                                f"blocked_fixed={fixed_recency_blocked_count} "
                                f"blocked_wrist={wrist_recency_blocked_count} "
                                f"blocked_fixed_ratio={fixed_recency_blocked_ratio:.2f}% "
                                f"blocked_wrist_ratio={wrist_recency_blocked_ratio:.2f}% "
                                f"candidates_after={len(candidates)}",
                                log_file,
                            )

                        # 이후 debug/reset에서 쓰기 위해 실제 blocked candidates만 저장
                        recency_blocked = recency_blocked_candidates
                    # ──────────────────────────────────────────────────────────
                   
                    # reuse_mode에 따라 필터링
                    if cfg.reuse_mode == "none_prune_off":
                        final_reusable_indices = []
                    elif cfg.reuse_mode == "wrist_only":
                        final_reusable_indices = [
                            idx for idx in candidates
                            if wrist_token_start <= idx < wrist_token_start + num_patches_per_image
                        ]
                    elif cfg.reuse_mode == "fixed_only":
                        final_reusable_indices = [
                            idx for idx in candidates
                            if fixed_token_start <= idx < fixed_token_start + num_patches_per_image
                        ]
                    else:  # "both"
                        fixed_cands = [
                            idx for idx in candidates
                            if fixed_token_start <= idx < fixed_token_start + num_patches_per_image
                        ]
                        wrist_cands = [
                            idx for idx in candidates
                            if wrist_token_start <= idx < wrist_token_start + num_patches_per_image
                        ]

                        # ── 기존 view-aware budget cap 유지 ─────────────────────
                        fixed_cands = apply_view_budget_cap(
                            fixed_cands,
                            last_caches.get(prev_fixed_attention_key),
                            fixed_token_start,
                            num_patches_per_image,
                            cfg.fixed_max_reuse_ratio,
                        )
                        wrist_cands = apply_view_budget_cap(
                            wrist_cands,
                            last_caches.get(prev_wrist_attention_key),
                            wrist_token_start,
                            num_patches_per_image,
                            cfg.wrist_max_reuse_ratio,
                        )
                        # ───────────────────────────────────────────────────────

                        if cfg.use_adaptive_pruning:
                            pruning_layers = (
                                tuple(int(x) for x in cfg.progressive_pruning_layers)
                                if cfg.progressive_pruning_layers is not None
                                else (2, 6, 10)
                            )

                            if pruning_layers != (2, 6, 10):
                                raise ValueError(
                                    f"Global adaptive pruning v2 expects layers=(2,6,10), got {pruning_layers}"
                                )

                            # ------------------------------------------------------------
                            # Adaptive phase signal 계산
                            # ------------------------------------------------------------
                            if prev_signal_T_w_c is not None and T_w_c_curr is not None:
                                curr_cam_pos = np.asarray(T_w_c_curr[:3, 3], dtype=np.float32)
                                prev_cam_pos = np.asarray(prev_signal_T_w_c[:3, 3], dtype=np.float32)

                                adaptive_cam_trans_delta = float(np.linalg.norm(curr_cam_pos - prev_cam_pos))
                                adaptive_cam_rot_delta_rad = _rotation_delta_angle_rad(prev_signal_T_w_c, T_w_c_curr)
                            else:
                                adaptive_cam_trans_delta = None
                                adaptive_cam_rot_delta_rad = None

                            adaptive_recent_eef_speed = (
                                float(max(recent_eef_speeds))
                                if len(recent_eef_speeds) > 0
                                else 0.0
                            )

                            adaptive_recent_dgripper_asym = (
                                float(max(abs(x) for x in recent_dgripper_asyms))
                                if len(recent_dgripper_asyms) > 0
                                else 0.0
                            )

                            adaptive_recent_speed_drop_ratio = (
                                float(max(recent_speed_drop_ratios))
                                if len(recent_speed_drop_ratios) > 0
                                else 0.0
                            )

                            adaptive_recent_abs_accel = (
                                float(max(recent_abs_accels))
                                if len(recent_abs_accels) > 0
                                else 0.0
                            )

                            adaptive_recent_abs_jerk = (
                                float(max(recent_abs_jerks))
                                if len(recent_abs_jerks) > 0
                                else 0.0
                            )

                            adaptive_recent_action_eef_ratio = (
                                float(max(recent_action_eef_ratios))
                                if len(recent_action_eef_ratios) > 0
                                else 0.0
                            )

                            finite_direction_cosines = [
                                float(c)
                                for c in recent_direction_cosines
                                if np.isfinite(c)
                            ]

                            adaptive_direction_dip_count = int(
                                sum(1 for c in finite_direction_cosines if c < 0.3)
                            )

                            adaptive_direction_dip_ratio = (
                                float(adaptive_direction_dip_count) / max(1, len(finite_direction_cosines))
                                if len(finite_direction_cosines) > 0
                                else 0.0
                            )

                            adaptive_direction_cosine_std = (
                                float(np.nanstd(finite_direction_cosines))
                                if len(finite_direction_cosines) > 1
                                else 0.0
                            )

                            adaptive_direction_cosine_std = (
                                float(np.nanstd(list(recent_direction_cosines)))
                                if len(recent_direction_cosines) > 0
                                else 0.0
                            )

                            # ------------------------------------------------------------
                            # Online adaptive risk signals
                            # ------------------------------------------------------------
                            adaptive_signals = {
                                # warp / camera motion
                                "hole_ratio": hole_ratio,
                                "cam_trans_delta": adaptive_cam_trans_delta,
                                "cam_rot_delta_rad": adaptive_cam_rot_delta_rad,

                                # EEF absolute speed는 debug용. high speed 자체를 risk로 보지는 않음.
                                "recent_eef_speed": adaptive_recent_eef_speed,

                                # transition / contact 후보
                                "recent_speed_drop_ratio": adaptive_recent_speed_drop_ratio,
                                "recent_abs_accel": adaptive_recent_abs_accel,
                                "recent_abs_jerk": adaptive_recent_abs_jerk,
                                "recent_dgripper_asym": adaptive_recent_dgripper_asym,
                                "recent_action_eef_ratio": adaptive_recent_action_eef_ratio,
                                "direction_dip_ratio": adaptive_direction_dip_ratio,
                                "direction_cosine_std": adaptive_direction_cosine_std,
                            }

                            fixed_age_pre = _cache_age_stats(
                                patch_stale_count,
                                fixed_token_start,
                                num_patches_per_image,
                                stale_threshold=cfg.stale_force_threshold,
                            )

                            wrist_age_pre = _cache_age_stats(
                                patch_stale_count,
                                wrist_token_start,
                                num_patches_per_image,
                                stale_threshold=cfg.stale_force_threshold,
                            )

                            adaptive_exposure = {
                                # 현재 LLM forward 전에 이미 지나간 LLM call 수
                                "past_llm_calls": max(0, int(llm_call_count)),

                                "stale_force_threshold": int(cfg.stale_force_threshold),

                                "fixed_cache_age_mean": float(fixed_age_pre["mean"]),
                                "wrist_cache_age_mean": float(wrist_age_pre["mean"]),

                                "fixed_cache_age_max": float(fixed_age_pre["max_age"]),
                                "wrist_cache_age_max": float(wrist_age_pre["max_age"]),

                                "fixed_stale_ge_ratio": float(fixed_age_pre["ge_cap_ratio"]),
                                "wrist_stale_ge_ratio": float(wrist_age_pre["ge_cap_ratio"]),

                                "fixed_recency_blocked_ratio": float(fixed_recency_blocked_ratio),
                                "wrist_recency_blocked_ratio": float(wrist_recency_blocked_ratio),
                            }

                            # ------------------------------------------------------------
                            # normattn_global ordering 유지
                            # fixed/wrist count를 따로 강제하지 않음
                            # ------------------------------------------------------------
                            final_reusable_indices = order_candidates_for_pruning(
                                fixed_candidates=fixed_cands,
                                wrist_candidates=wrist_cands,
                                fixed_attn_map=last_caches.get(prev_fixed_attention_key),
                                wrist_attn_map=last_caches.get(prev_wrist_attention_key),
                                fixed_token_start=fixed_token_start,
                                wrist_token_start=wrist_token_start,
                                num_patches_per_image=num_patches_per_image,
                                mode=cfg.candidate_ordering_mode,
                                progressive_drop_ratios=list(cfg.progressive_drop_ratios) if cfg.progressive_drop_ratios else None,
                            )

                            # ------------------------------------------------------------
                            # reference full token count
                            # ------------------------------------------------------------
                            llama_model = model.language_model.model
                            prev_forward_cache = getattr(llama_model, "last_forward_cache", None)

                            if prev_forward_cache is not None and hasattr(prev_forward_cache, "get_seq_length"):
                                ref_total_tokens = int(prev_forward_cache.get_seq_length())
                            else:
                                q_total = (getattr(llama_model, "last_q_ratios", {}) or {}).get("total", {})
                                ref_total_tokens = int(q_total.get("original_seq_length", 600))

                            # ------------------------------------------------------------
                            # Online distribution-normalized adaptive pruning
                            # ------------------------------------------------------------
                            (
                                adaptive_phase_info,
                                target_total_ratios,
                                adaptive_progressive_drop_ratios,
                                adaptive_target_counts,
                                adaptive_pruning_stats,
                            ) = build_online_global_adaptive_pruning_plan(
                                risk_controller=adaptive_risk_controller,
                                signals=adaptive_signals,
                                exposure=adaptive_exposure,
                                n_candidates=len(final_reusable_indices),
                                ref_total_tokens=ref_total_tokens,
                                fixed_available=len(fixed_cands),
                                wrist_available=len(wrist_cands),
                                min_total_prune_ratios=cfg.adaptive_min_total_prune_ratios,
                                max_total_prune_ratios=cfg.adaptive_max_total_prune_ratios,
                            )

                            global_phase = adaptive_phase_info["global_phase"]

                            if log_file is not None:
                                log_message(
                                    "[ADAPTIVE PRUNING ONLINE] "
                                    f"step={t:03d} "
                                    f"llm_call={next_llm_call_count} "
                                    f"global_phase={global_phase} "
                                    f"budget_mode={adaptive_phase_info.get('budget_mode', 'NA')} "
                                    f"risk_score={adaptive_phase_info['risk_score']:.4f} "
                                    f"instant_risk_score={adaptive_phase_info['instant_risk_score']:.4f} "
                                    f"dominant_signal={adaptive_phase_info['dominant_signal']} "
                                    f"per_signal_risk={adaptive_phase_info.get('per_signal_risk', {})} "
                                    f"score_components={adaptive_phase_info.get('score_components', {})} "
                                    f"exposure_components={adaptive_phase_info.get('exposure_components', {})} "
                                    f"history_lengths={adaptive_phase_info['history_lengths']} "
                                    f"signals={adaptive_phase_info['signals']} "
                                    f"exposure={adaptive_exposure} "
                                    f"fixed_avail={len(fixed_cands)} "
                                    f"wrist_avail={len(wrist_cands)} "
                                    f"num_candidates={len(final_reusable_indices)} "
                                    f"ref_total_tokens={ref_total_tokens} "
                                    f"target_total_ratios={target_total_ratios} "
                                    f"target_counts={adaptive_target_counts} "
                                    f"dynamic_ratios={adaptive_progressive_drop_ratios}",
                                    log_file,
                                )

                            # # ------------------------------------------------------------
                            # # Global risk phase 결정
                            # # candidate count / recency_blocked_count는 여기서 쓰지 않음
                            # # ------------------------------------------------------------
                            # adaptive_phase_info = classify_global_adaptive_pruning_phase(
                            #     hole_ratio=hole_ratio,
                            #     cam_trans_delta=adaptive_cam_trans_delta,
                            #     cam_rot_delta_rad=adaptive_cam_rot_delta_rad,
                            #     recent_eef_speed=adaptive_recent_eef_speed,
                            #     recent_dgripper_asym=adaptive_recent_dgripper_asym,
                            #     hole_high_thr=adaptive_signal_thresholds["hole_ratio"],
                            #     cam_trans_high_thr=adaptive_signal_thresholds["cam_trans_delta"],
                            #     cam_rot_high_thr=adaptive_signal_thresholds["cam_rot_delta_rad"],
                            #     eef_speed_high_thr=adaptive_signal_thresholds["recent_eef_speed"],
                            #     dgripper_asym_high_thr=adaptive_signal_thresholds["recent_dgripper_asym"],
                            # )

                            # adaptive_phase_info.setdefault(
                            #     "signals",
                            #     {
                            #         "hole_ratio": hole_ratio,
                            #         "cam_trans_delta": adaptive_cam_trans_delta,
                            #         "cam_rot_delta_rad": adaptive_cam_rot_delta_rad,
                            #         "recent_eef_speed": adaptive_recent_eef_speed,
                            #         "recent_dgripper_asym": adaptive_recent_dgripper_asym,
                            #     },
                            # )

                            # global_phase = adaptive_phase_info["global_phase"]

                            # # ------------------------------------------------------------
                            # # normattn_global ordering 유지
                            # # fixed/wrist count를 따로 강제하지 않음
                            # # ------------------------------------------------------------
                            # final_reusable_indices = order_candidates_for_pruning(
                            #     fixed_candidates=fixed_cands,
                            #     wrist_candidates=wrist_cands,
                            #     fixed_attn_map=last_caches.get(prev_fixed_attention_key),
                            #     wrist_attn_map=last_caches.get(prev_wrist_attention_key),
                            #     fixed_token_start=fixed_token_start,
                            #     wrist_token_start=wrist_token_start,
                            #     num_patches_per_image=num_patches_per_image,
                            #     mode=cfg.candidate_ordering_mode,
                            #     progressive_drop_ratios=list(cfg.progressive_drop_ratios) if cfg.progressive_drop_ratios else None,
                            # )

                            # # ------------------------------------------------------------
                            # # 전체 token 기준 target ratio를 candidate-list 기준 ratio로 변환
                            # # ------------------------------------------------------------
                            # target_total_ratios = GLOBAL_TOTAL_RATIO_TARGETS[global_phase]
                            # llama_model = model.language_model.model
                            # prev_forward_cache = getattr(llama_model, "last_forward_cache", None)

                            # if prev_forward_cache is not None and hasattr(prev_forward_cache, "get_seq_length"):
                            #     ref_total_tokens = int(prev_forward_cache.get_seq_length())
                            # else:
                            #     q_total = (getattr(llama_model, "last_q_ratios", {}) or {}).get("total", {})
                            #     ref_total_tokens = int(q_total.get("original_seq_length", 600))

                            # (
                            #     adaptive_progressive_drop_ratios,
                            #     adaptive_target_counts,
                            # ) = total_ratio_targets_to_candidate_ratios(
                            #     target_total_ratios=target_total_ratios,
                            #     n_candidates=len(final_reusable_indices),
                            #     ref_total_tokens=ref_total_tokens,
                            # )

                            # adaptive_pruning_stats = {
                            #     "global_phase": global_phase,
                            #     "risk_score": adaptive_phase_info["risk_score"],
                            #     "reasons": adaptive_phase_info["reasons"],
                            #     "fixed_reasons": adaptive_phase_info["fixed_reasons"],
                            #     "wrist_reasons": adaptive_phase_info["wrist_reasons"],
                            #     "target_total_ratios": target_total_ratios,
                            #     "target_counts": adaptive_target_counts,
                            #     "dynamic_ratios": adaptive_progressive_drop_ratios,
                            #     "num_candidates": len(final_reusable_indices),
                            #     "ref_total_tokens": ref_total_tokens,
                            #     "fixed_available": len(fixed_cands),
                            #     "wrist_available": len(wrist_cands),
                            # }

                            # if log_file is not None:
                            #     log_message(
                            #         "[ADAPTIVE PRUNING ONLINE] "
                            #         f"step={t:03d} "
                            #         f"llm_call={next_llm_call_count} "
                            #         f"global_phase={global_phase} "
                            #         f"risk_score={adaptive_phase_info['risk_score']:.4f} "
                            #         f"instant_risk_score={adaptive_phase_info['instant_risk_score']:.4f} "
                            #         f"dominant_signal={adaptive_phase_info['dominant_signal']} "
                            #         f"per_signal_risk={adaptive_phase_info['per_signal_risk']} "
                            #         f"history_lengths={adaptive_phase_info['history_lengths']} "
                            #         f"signals={adaptive_phase_info['signals']} "
                            #         f"fixed_avail={len(fixed_cands)} "
                            #         f"wrist_avail={len(wrist_cands)} "
                            #         f"num_candidates={len(final_reusable_indices)} "
                            #         f"ref_total_tokens={ref_total_tokens} "
                            #         f"target_total_ratios={target_total_ratios} "
                            #         f"target_counts={adaptive_target_counts} "
                            #         f"dynamic_ratios={adaptive_progressive_drop_ratios}",
                            #         log_file,
                            #     )

                        else:
                            if cfg.candidate_ordering_mode == "original":
                                final_reusable_indices = candidates  # 기존 동작 그대로
                            else:
                                final_reusable_indices = order_candidates_for_pruning(
                                    fixed_candidates=fixed_cands,
                                    wrist_candidates=wrist_cands,
                                    fixed_attn_map=last_caches.get(prev_fixed_attention_key),
                                    wrist_attn_map=last_caches.get(prev_wrist_attention_key),
                                    fixed_token_start=fixed_token_start,
                                    wrist_token_start=wrist_token_start,
                                    num_patches_per_image=num_patches_per_image,
                                    mode=cfg.candidate_ordering_mode,
                                    progressive_drop_ratios=list(cfg.progressive_drop_ratios) if cfg.progressive_drop_ratios else None,
                                )
    
                # keyframe step에는 재사용 패치 없음
                applied_reusable_indices = None if hard_refresh_step else final_reusable_indices    
                # model config에 reusable_patches 주입
                if cfg.disable_kv_cache_reuse or not applied_reusable_indices:
                    model.language_model.config.reusable_patches = None

                    # adaptive ratio가 이전 forward에서 남아있지 않게 정리
                    if cfg.use_adaptive_pruning:
                        model.language_model.config.progressive_drop_ratios = None

                else:
                    indices_tensor = torch.tensor(
                        applied_reusable_indices,
                        dtype=torch.long,
                        device=DEVICE,
                    )
                    model.language_model.config.reusable_patches = indices_tensor  # v_token_start=1

                    if cfg.use_adaptive_pruning:
                        pruning_layers = (
                            list(cfg.progressive_pruning_layers)
                            if cfg.progressive_pruning_layers is not None
                            else [2, 6, 10]
                        )

                        if adaptive_progressive_drop_ratios is None:
                            # candidate가 0개였거나 adaptive plan 생성 실패한 경우
                            model.language_model.config.progressive_drop_ratios = None
                        else:
                            model.language_model.config.progressive_pruning_layers = pruning_layers
                            model.language_model.config.progressive_drop_ratios = [
                                float(x) for x in adaptive_progressive_drop_ratios
                            ]

                    else:
                        # 기존 방식
                        model.language_model.config.progressive_pruning_layers = (
                            list(cfg.progressive_pruning_layers)
                            if cfg.progressive_pruning_layers
                            else None
                        )
                        model.language_model.config.progressive_drop_ratios = (
                            [float(x) for x in cfg.progressive_drop_ratios]
                            if cfg.progressive_drop_ratios
                            else None
                        )

                
                    # # ── 여기 ──
                    # wrist_indices = [idx for idx in indices_tensor.tolist() if 257 <= idx < 513]
                    # non_wrist_indices = [idx for idx in indices_tensor.tolist() if not (257 <= idx < 513)]

                    # if wrist_indices:
                    #     reuse_pos = wrist_indices[0]   # wrist drop될 position
                    #     kept_pos = non_wrist_indices[0] if non_wrist_indices else wrist_indices[-1] + 1  # kept position
                    #     print(f"reuse_pos={reuse_pos} (wrist), kept_pos={kept_pos}")
                    # else:
                    #     reuse_pos = None
                    #     kept_pos = None
                    # # ──────────
    
                reusable_count = 0 if applied_reusable_indices is None else len(applied_reusable_indices)
                fixed_reusable_count, wrist_reusable_count = _count_reusable_tokens_by_view(
                    applied_reusable_indices,
                    fixed_token_start,
                    wrist_token_start,
                    num_patches_per_image,
                )
                # log_message(f"[REUSE DBG] step={t:03d} reusable_patch_count={reusable_count}", log_file)

            else:
                model.language_model.config.reusable_patches = None
                warped_caches = None
 
            # -----------------------------------------------------------------
            # [Step 4] Observation 준비 + 모델 추론
            #
            # oft 고유 방식: action_queue가 비었을 때만 모델을 호출하고
            # warped_cache / last_caches를 함께 넘겨 KV cache를 재사용.
            # -----------------------------------------------------------------

            if cfg.use_dynam_cache and will_call_llm:
                if hard_refresh_step:
                    fixed_reusable_patch_indices = set()
                    wrist_reusable_patch_indices = set()
                    fixed_critical_patch_indices = set()
                    wrist_critical_patch_indices = set()

                    last_heatmap_fixed = make_patch_overlay(
                        fixed_img,
                        reusable_patch_indices=fixed_reusable_patch_indices,
                        critical_patch_indices=fixed_critical_patch_indices,
                        num_patches_per_image=num_patches_per_image,
                    )

                    last_heatmap_wrist = make_patch_overlay(
                        wrist_img,
                        reusable_patch_indices=wrist_reusable_patch_indices,
                        critical_patch_indices=wrist_critical_patch_indices,
                        num_patches_per_image=num_patches_per_image,
                    )

                else:
                    fixed_reusable_patch_indices = _token_indices_to_patch_indices(
                        applied_reusable_indices,
                        fixed_token_start,
                        num_patches_per_image,
                    )
                    wrist_reusable_patch_indices = _token_indices_to_patch_indices(
                        applied_reusable_indices,
                        wrist_token_start,
                        num_patches_per_image,
                    )
                    fixed_critical_patch_indices = _token_indices_to_patch_indices(
                        fixed_task_relevant_indices,
                        fixed_token_start,
                        num_patches_per_image,
                    )
                    wrist_critical_patch_indices = _token_indices_to_patch_indices(
                        wrist_task_relevant_indices,
                        wrist_token_start,
                        num_patches_per_image,
                    )

                    last_heatmap_fixed = make_patch_overlay(
                        fixed_img,
                        reusable_patch_indices=fixed_reusable_patch_indices,
                        critical_patch_indices=fixed_critical_patch_indices,
                        num_patches_per_image=num_patches_per_image,
                    )

                    last_heatmap_wrist = make_patch_overlay(
                        wrist_img,
                        reusable_patch_indices=wrist_reusable_patch_indices,
                        critical_patch_indices=wrist_critical_patch_indices,
                        num_patches_per_image=num_patches_per_image,
                    )


            elif not cfg.use_dynam_cache:
                # dynam_cache 없는 경우: 원본 이미지 자체를 캐싱
                last_heatmap_fixed = fixed_img
                last_heatmap_wrist = wrist_img

            # will_call_llm 여부와 무관하게 항상 append
            replay_images_heatmap.append(last_heatmap_fixed)
            replay_images_wrist_heatmap.append(last_heatmap_wrist)

            # ── ADAPTIVE PHASE DEBUG IMAGE ─────────────────────────
            if (
                cfg.use_dynam_cache
                and will_call_llm
                and cfg.use_adaptive_pruning
                and adaptive_phase_info is not None
            ):
                adaptive_debug_path = save_adaptive_phase_debug_frame(
                    save_dir=adaptive_phase_debug_dir,
                    step=t,
                    llm_call=next_llm_call_count,
                    task_description=task_description,
                    fixed_img=fixed_img,
                    wrist_img=wrist_img,
                    fixed_overlay_img=last_heatmap_fixed,
                    wrist_overlay_img=last_heatmap_wrist,
                    adaptive_phase_info=adaptive_phase_info,
                    adaptive_pruning_stats=adaptive_pruning_stats,
                    adaptive_thresholds=None,
                    target_total_ratios=target_total_ratios,
                    adaptive_target_counts=adaptive_target_counts,
                    adaptive_progressive_drop_ratios=adaptive_progressive_drop_ratios,
                    fixed_critical_count=len(fixed_task_relevant_indices),
                    wrist_critical_count=len(wrist_task_relevant_indices),
                    fixed_reusable_count=fixed_reusable_count,
                    wrist_reusable_count=wrist_reusable_count,
                    num_patches_per_image=num_patches_per_image,
                )

                log_message(
                    f"[ADAPTIVE PHASE DEBUG IMAGE] step={t:03d} "
                    f"llm_call={next_llm_call_count} "
                    f"path={adaptive_debug_path}",
                    log_file,
                )
            # ───────────────────────────────────────────────────────


            # If action queue is empty, requery model
            if will_call_llm:
                chunk_wall_start = step_start_time
                _t_model_start = time.perf_counter()

                if cfg.use_dynam_cache:
                    llm_call_count += 1

                    if not hard_refresh_step:
                        perf_counters["llm_reuse_calls"] += 1
                        perf_counters["llm_reusable_fixed_tokens"] += fixed_reusable_count
                        perf_counters["llm_reusable_wrist_tokens"] += wrist_reusable_count

                    # Query model to get action

                    # ── clear stale exact-reuse debug before current forward ─────────────
                    llama_model = model.language_model.model
                    llama_model.last_reused_patch_indices = None
                    llama_model.last_reused_patch_step_layer = None
                    llama_model.last_progressive_debug = None
                    llama_model.last_q_ratios = {}
                    # ───────────────────────────────────────────────────────────────────

                    log_message(
                        f"[PREFWD REUSE CHECK] step={t:03d} "
                        f"llm_call={llm_call_count} "
                        f"is_keyframe={is_keyframe_step} "
                        f"force_keyframe={force_keyframe_refresh} "
                        f"hard_refresh={hard_refresh_step} "
                        f"use_recency_cap={cfg.use_recency_cap} "
                        f"reusable_patches_none={model.language_model.config.reusable_patches is None} "
                        f"warped_caches_none={warped_caches is None} "
                        f"keyframe_cache_none={keyframe_cache is None} "
                        f"cache_anchor_cache_none={cache_anchor_cache is None} "
                        f"applied_reusable_len={0 if applied_reusable_indices is None else len(applied_reusable_indices)}",
                        log_file,
                    )
                    
                    # run_libero_eval.py, get_action() 호출 직전
                    actions = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                        last_caches=last_caches,
                        warped_cache=None if hard_refresh_step else warped_caches,
                    )

                else:
                    # Plain OpenVLA-OFT path: no warp, no reuse patch, no cache side-channel.
                    actions = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )
                _t_model_end = time.perf_counter()

                # # ── VISUALIZE TEXT-AWARE COMPARISON ─────────────────────────
                # if cfg.use_text_aware_debug and not force_keyframe_refresh:
                #     fixed_mixed_map = last_caches.get("latest_fixed_spatial_map")
                #     wrist_mixed_map = last_caches.get("latest_wrist_spatial_map")
                #     fixed_text_only_map = last_caches.get("latest_fixed_text_only_map")
                #     wrist_text_only_map = last_caches.get("latest_wrist_text_only_map")

                #     if fixed_text_only_map is not None and wrist_text_only_map is not None:
                #         text_aware_path = save_text_aware_comparison_grid(
                #             save_dir=text_aware_debug_dir,
                #             step=t,
                #             llm_call=llm_call_count,
                #             fixed_image=fixed_img,
                #             wrist_image=wrist_img,
                #             fixed_mixed_map=fixed_mixed_map,
                #             wrist_mixed_map=wrist_mixed_map,
                #             fixed_text_only_map=fixed_text_only_map,
                #             wrist_text_only_map=wrist_text_only_map,
                #             colormap="turbo",
                #             show_grid_lines=True,
                #         )
                #         log_message(
                #             f"[TEXT-AWARE VIS] step={t:03d} llm_call={llm_call_count} path={text_aware_path}",
                #             log_file,
                #         )
                # # ── VISUALIZE TEXT-AWARE COMPARISON ─────────────────────────

                # # ── VISUALIZE TEXT-AWARE 3-WAY COMPARISON ─────────────────────────
                # if cfg.use_stopword_filtered_debug and not force_keyframe_refresh:
                #     fixed_mixed_map = last_caches.get("latest_fixed_spatial_map")
                #     wrist_mixed_map = last_caches.get("latest_wrist_spatial_map")

                #     fixed_text_only_map = last_caches.get("latest_fixed_text_only_map")
                #     wrist_text_only_map = last_caches.get("latest_wrist_text_only_map")

                #     fixed_stopword_map = last_caches.get("latest_fixed_stopword_filtered_map")
                #     wrist_stopword_map = last_caches.get("latest_wrist_stopword_filtered_map")

                #     log_message(
                #         f"[TEXT-AWARE 3WAY CHECK] step={t:03d} "
                #         f"mixed=({fixed_mixed_map is not None},{wrist_mixed_map is not None}) "
                #         f"text=({fixed_text_only_map is not None},{wrist_text_only_map is not None}) "
                #         f"stopword=({fixed_stopword_map is not None},{wrist_stopword_map is not None})",
                #         log_file,
                #     )

                #     if (
                #         fixed_mixed_map is not None
                #         and wrist_mixed_map is not None
                #         and fixed_text_only_map is not None
                #         and wrist_text_only_map is not None
                #         and fixed_stopword_map is not None
                #         and wrist_stopword_map is not None
                #     ):
                #         three_way_path = save_text_aware_three_way_comparison_grid(
                #             save_dir=text_aware_debug_dir,
                #             step=t,
                #             llm_call=llm_call_count,
                #             fixed_image=fixed_img,
                #             wrist_image=wrist_img,
                #             fixed_mixed_map=fixed_mixed_map,
                #             wrist_mixed_map=wrist_mixed_map,
                #             fixed_text_only_map=fixed_text_only_map,
                #             wrist_text_only_map=wrist_text_only_map,
                #             fixed_stopword_filtered_map=fixed_stopword_map,
                #             wrist_stopword_filtered_map=wrist_stopword_map,
                #             colormap="turbo",
                #             show_grid_lines=True,
                #         )

                #         log_message(
                #             f"[TEXT-AWARE 3WAY VIS] step={t:03d} "
                #             f"llm_call={llm_call_count} path={three_way_path}",
                #             log_file,
                #         )
                # # ── VISUALIZE TEXT-AWARE 3-WAY COMPARISON ─────────────────────────

                # # ── VISUALIZE QUERY-MODE ATTENTION MAPS ─────────────────────────
                # if cfg.use_query_mode_map_debug and not force_keyframe_refresh:
                #     fixed_mixed_map = last_caches.get("latest_fixed_spatial_map")
                #     wrist_mixed_map = last_caches.get("latest_wrist_spatial_map")

                #     fixed_text_only_map = last_caches.get("latest_fixed_text_only_map")
                #     wrist_text_only_map = last_caches.get("latest_wrist_text_only_map")

                #     fixed_content_words_map = last_caches.get("latest_fixed_content_words_map")
                #     wrist_content_words_map = last_caches.get("latest_wrist_content_words_map")

                #     fixed_status_only_map = last_caches.get("latest_fixed_status_only_map")
                #     wrist_status_only_map = last_caches.get("latest_wrist_status_only_map")

                #     fixed_action_only_map = last_caches.get("latest_fixed_action_only_map")
                #     wrist_action_only_map = last_caches.get("latest_wrist_action_only_map")

                #     log_message(
                #         f"[QUERY-MODE VIS CHECK] step={t:03d} llm_call={llm_call_count} "
                #         f"mixed=({fixed_mixed_map is not None},{wrist_mixed_map is not None}) "
                #         f"text=({fixed_text_only_map is not None},{wrist_text_only_map is not None}) "
                #         f"content=({fixed_content_words_map is not None},{wrist_content_words_map is not None}) "
                #         f"status=({fixed_status_only_map is not None},{wrist_status_only_map is not None}) "
                #         f"action=({fixed_action_only_map is not None},{wrist_action_only_map is not None})",
                #         log_file,
                #     )

                #     if (
                #         fixed_mixed_map is not None
                #         and wrist_mixed_map is not None
                #         and fixed_text_only_map is not None
                #         and wrist_text_only_map is not None
                #         and fixed_content_words_map is not None
                #         and wrist_content_words_map is not None
                #         and fixed_action_only_map is not None
                #         and wrist_action_only_map is not None
                #     ):
                #         query_mode_path = save_query_mode_attention_grid(
                #             save_dir=text_aware_debug_dir,
                #             step=t,
                #             llm_call=llm_call_count,
                #             fixed_image=fixed_img,
                #             wrist_image=wrist_img,
                #             fixed_mixed_map=fixed_mixed_map,
                #             wrist_mixed_map=wrist_mixed_map,
                #             fixed_text_only_map=fixed_text_only_map,
                #             wrist_text_only_map=wrist_text_only_map,
                #             fixed_content_words_map=fixed_content_words_map,
                #             wrist_content_words_map=wrist_content_words_map,
                #             fixed_status_only_map=fixed_status_only_map,
                #             wrist_status_only_map=wrist_status_only_map,
                #             fixed_action_only_map=fixed_action_only_map,
                #             wrist_action_only_map=wrist_action_only_map,
                #             colormap="turbo",
                #             show_grid_lines=True,
                #         )

                #         log_message(
                #             f"[QUERY-MODE VIS] step={t:03d} "
                #             f"llm_call={llm_call_count} path={query_mode_path}",
                #             log_file,
                #         )
                # # ── VISUALIZE QUERY-MODE ATTENTION MAPS ─────────────────────────

                # -----------------------------------------------------------------
                # [Visualize] Attention Heat map 
                # -----------------------------------------------------------------
                # if cfg.use_dynam_cache:
                    # save_attention_heatmap_grid(
                    #     save_dir=attention_heatmap_dir,
                    #     step=t,
                    #     fixed_image=fixed_img,
                    #     wrist_image=wrist_img,
                    #     fixed_attention_map=last_caches.get("latest_fixed_spatial_map"),
                    #     wrist_attention_map=last_caches.get("latest_wrist_spatial_map"),
                    #     fixed_final_layer_attention_map=last_caches.get("latest_fixed_final_layer_spatial_map"),
                    #     wrist_final_layer_attention_map=last_caches.get("latest_wrist_final_layer_spatial_map"),
                    #     fixed_patch_overlay=last_heatmap_fixed,
                    #     wrist_patch_overlay=last_heatmap_wrist,
                    #     alpha=0.45,
                    # )

                    # per_layer_maps = {}
                    # for _lid in range(32):
                    #     _fk = f"layer_{_lid:02d}_fixed_map"
                    #     _wk = f"layer_{_lid:02d}_wrist_map"
                    #     if last_caches.get(_fk) is not None:
                    #         per_layer_maps[_lid] = (last_caches[_fk], last_caches.get(_wk))

                    # if per_layer_maps:
                    #     save_all_layers_heatmap_grid(
                    #         save_dir=attention_heatmap_dir,
                    #         step=t,
                    #         fixed_image=fixed_img,
                    #         wrist_image=wrist_img,
                    #         per_layer_maps=per_layer_maps,
                    #         alpha=0.45,
                    #     )
                    
                action_queue.extend(actions)

                current_chunk_inference_s = _t_model_end - _t_model_start
                current_chunk_env_s = 0.0
                current_chunk_actions_executed = 0
                current_chunk_size = len(actions)


                # # ── 토큰 구조 디버그 (첫 번째 LLM 호출에서만 1회 출력) ──────────
                # if cfg.use_dynam_cache and llm_call_count == 1:
                #     llama_model = model.language_model.model
                #     cache = getattr(llama_model, "last_forward_cache", None)
                #     kept = getattr(llama_model, "last_forward_kept_query_positions", None)

                #     if cache is not None:
                #         total_cache_len = cache.get_seq_length()
                #         print(f"\n{'='*60}")
                #         print(f"[TOKEN STRUCTURE DEBUG]")
                #         print(f"  total cache seq_len     : {total_cache_len}")
                #         print(f"  inputs_embeds shape     : prefill seq_len = {total_cache_len}")
                #         print(f"")
                #         print(f"  fixed_token_start       : {fixed_token_start}")
                #         print(f"  fixed_token_end         : {fixed_token_start + num_patches_per_image}")
                #         print(f"  wrist_token_start       : {wrist_token_start}")
                #         print(f"  wrist_token_end         : {wrist_token_start + num_patches_per_image}")
                #         print(f"")
                #         # instruction + status 추정
                #         text_token_start = wrist_token_start + num_patches_per_image
                #         if cfg.use_proprio:
                #             text_token_start += 1  # proprio projector 1토큰
                #         print(f"  text_token_start (est)  : {text_token_start}")
                #         print(f"  text_token_count (est)  : {total_cache_len - text_token_start}")
                #         print(f"")
                #         if kept is not None:
                #             print(f"  kept_query_positions    : {kept.tolist()}")
                #             print(f"  kept_query_positions max: {kept.max().item()}")
                #             print(f"  kept_query_positions min: {kept.min().item()}")
                #             print(f"  kept_query count        : {kept.numel()}")
                #         print(f"{'='*60}\n")
                # ────────────────────────────────────────────────────────────────
                if cfg.use_dynam_cache:
                    prev_attn_T_w_c = T_w_c_curr
                    prev_attn_wrist_img = wrist_img
                    prev_llm_fixed_img = fixed_img


                # LLaMA 성능 지표 로깅 (실제 추론이 발생한 step에서만)
                _log_llama_perf_metrics(log_file, model, perf_counters=perf_counters)
                        
                llama_model = model.language_model.model
                progressive_debug = getattr(llama_model, "last_progressive_debug", None)

                # # ── VISUALIZE REUSE PATCH ─────────────────────────
                exact_reused_set = set()

                # stale distribution / overlap defaults
                fixed_stale_reuse_ratio = 0.0
                wrist_stale_reuse_ratio = 0.0
                fixed_stale_critical_overlap_ratio = 0.0
                wrist_stale_critical_overlap_ratio = 0.0
                fixed_stale_critical_overlap_count = 0
                wrist_stale_critical_overlap_count = 0
                fixed_stale_reuse_count = 0
                wrist_stale_reuse_count = 0

                if not hard_refresh_step:
                    _exact_reused_abs = getattr(llama_model, "last_reused_patch_indices", None)
                    if _exact_reused_abs is not None and _exact_reused_abs.numel() > 0:
                        _exact_reused_list = _exact_reused_abs.tolist()
                        _fixed_exact_patches = [
                            idx - fixed_token_start
                            for idx in _exact_reused_list
                            if fixed_token_start <= idx < fixed_token_start + num_patches_per_image
                        ]
                        _wrist_exact_patches = [
                            idx - wrist_token_start
                            for idx in _exact_reused_list
                            if wrist_token_start <= idx < wrist_token_start + num_patches_per_image
                        ]

                        log_message(
                            f"[EXACT REUSE DBG] step={t:03d} "
                            f"last_layer={getattr(llama_model, 'last_reused_patch_step_layer', None)} "
                            f"n_indices={_exact_reused_abs.numel()}",
                            log_file,
                        )

                        save_exact_reuse_grid(
                            save_dir=exact_reuse_dir,
                            step=t,
                            fixed_image=fixed_img,
                            wrist_image=wrist_img,
                            fixed_exact_reused_patch_indices=_fixed_exact_patches,
                            wrist_exact_reused_patch_indices=_wrist_exact_patches,
                            num_patches_per_image=num_patches_per_image,
                        )

                    else:
                        _exact_reused_list = []

                    exact_reused_set = set(int(x) for x in _exact_reused_list)

                    # ── stale duration update ─────────────────────────────────
                    # exact_reused_set = 이번 LLM forward에서 실제로 cache reuse/prune된 visual token들
                    # exact reuse된 patch만 stale count +1
                    # 나머지 모든 visual patch는 이번 forward에서 recompute된 것으로 보고 reset

                    all_visual_token_set = set(all_visual_token_indices)

                    for idx in all_visual_token_set:
                        if idx in exact_reused_set:
                            patch_stale_count[idx] = patch_stale_count.get(idx, 0) + 1
                        else:
                            patch_stale_count[idx] = 0
                    # ─────────────────────────────────────────────────────────

                    # ---------------------------------------------------------
                    # Stale reuse / stale-critical overlap snapshot
                    # critical patch를 reset하기 전에 봐야 overlap이 의미 있음.
                    # ---------------------------------------------------------
                    stale_tokens_snapshot = {
                        int(idx)
                        for idx, cnt in patch_stale_count.items()
                        if cnt >= cfg.stale_force_threshold
                    }

                    fixed_critical_set = set(int(x) for x in (fixed_task_relevant_indices or []))
                    wrist_critical_set = set(int(x) for x in (wrist_task_relevant_indices or []))

                    fixed_exact_reused_set = {
                        idx for idx in exact_reused_set
                        if fixed_token_start <= idx < fixed_token_start + num_patches_per_image
                    }
                    wrist_exact_reused_set = {
                        idx for idx in exact_reused_set
                        if wrist_token_start <= idx < wrist_token_start + num_patches_per_image
                    }

                    fixed_stale_reused_set = stale_tokens_snapshot & fixed_exact_reused_set
                    wrist_stale_reused_set = stale_tokens_snapshot & wrist_exact_reused_set

                    fixed_stale_reuse_count = len(fixed_stale_reused_set)
                    wrist_stale_reuse_count = len(wrist_stale_reused_set)

                    fixed_stale_reuse_ratio = (
                        fixed_stale_reuse_count / len(fixed_exact_reused_set) * 100
                        if len(fixed_exact_reused_set) > 0 else 0.0
                    )
                    wrist_stale_reuse_ratio = (
                        wrist_stale_reuse_count / len(wrist_exact_reused_set) * 100
                        if len(wrist_exact_reused_set) > 0 else 0.0
                    )

                    fixed_stale_critical_overlap_count = len(stale_tokens_snapshot & fixed_critical_set)
                    wrist_stale_critical_overlap_count = len(stale_tokens_snapshot & wrist_critical_set)

                    fixed_stale_critical_overlap_ratio = (
                        fixed_stale_critical_overlap_count / len(fixed_critical_set) * 100
                        if len(fixed_critical_set) > 0 else 0.0
                    )
                    wrist_stale_critical_overlap_ratio = (
                        wrist_stale_critical_overlap_count / len(wrist_critical_set) * 100
                        if len(wrist_critical_set) > 0 else 0.0
                    )
                    # ---------------------------------------------------------

                    # critical로 잡혀 처음부터 재계산 대상이던 patch도 리셋
                    for idx in (fixed_task_relevant_indices or []) + (wrist_task_relevant_indices or []):
                        patch_stale_count[idx] = 0

                    

                # ── CALL SIGNAL 로그 ────────────────────────────────────────
                fixed_critical_ratio = len(fixed_task_relevant_indices) / num_patches_per_image * 100
                wrist_critical_ratio = len(wrist_task_relevant_indices) / num_patches_per_image * 100

                reusable_candidate_count = 0 if applied_reusable_indices is None else len(applied_reusable_indices)
                exact_reuse_count = len(exact_reused_set)
                exact_reuse_ratio_of_candidates = (
                    exact_reuse_count / reusable_candidate_count * 100
                    if reusable_candidate_count > 0 else 0.0
                )

                # ── cache age stats ───────────────────────────────────────
                fixed_age_stats = _cache_age_stats(
                    patch_stale_count,
                    fixed_token_start,
                    num_patches_per_image,
                    stale_threshold=cfg.stale_force_threshold,
                )
                wrist_age_stats = _cache_age_stats(
                    patch_stale_count,
                    wrist_token_start,
                    num_patches_per_image,
                    stale_threshold=cfg.stale_force_threshold,
                )

                fixed_cache_age_max = fixed_age_stats["max_age"]
                wrist_cache_age_max = wrist_age_stats["max_age"]

                fixed_stale_eq_threshold_count = fixed_age_stats["eq_cap_count"]
                wrist_stale_eq_threshold_count = wrist_age_stats["eq_cap_count"]

                fixed_stale_ge_threshold_count = fixed_age_stats["ge_cap_count"]
                wrist_stale_ge_threshold_count = wrist_age_stats["ge_cap_count"]

                total_stale_eq_threshold_count = (
                    fixed_stale_eq_threshold_count + wrist_stale_eq_threshold_count
                )
                total_stale_ge_threshold_count = (
                    fixed_stale_ge_threshold_count + wrist_stale_ge_threshold_count
                )

                # ── STALE DBG: cache age stats 계산 후에 찍어야 함 ─────────────
                if patch_stale_count:
                    max_stale_idx = max(patch_stale_count, key=patch_stale_count.get)
                    max_stale_val = patch_stale_count[max_stale_idx]

                    log_message(
                        f"[STALE DBG] step={t:03d} llm_call={llm_call_count} "
                        f"exact_reused_count={len(exact_reused_set)} "
                        f"max_stale_patch_idx={max_stale_idx} "
                        f"max_stale_age={max_stale_val} "
                        f"fixed_eq{cfg.stale_force_threshold}_index_count={fixed_stale_eq_threshold_count} "
                        f"wrist_eq{cfg.stale_force_threshold}_index_count={wrist_stale_eq_threshold_count} "
                        f"total_eq{cfg.stale_force_threshold}_index_count={total_stale_eq_threshold_count} "
                        f"fixed_ge{cfg.stale_force_threshold}_index_count={fixed_stale_ge_threshold_count} "
                        f"wrist_ge{cfg.stale_force_threshold}_index_count={wrist_stale_ge_threshold_count} "
                        f"total_ge{cfg.stale_force_threshold}_index_count={total_stale_ge_threshold_count} "
                        f"tracked_patches={len(patch_stale_count)}",
                        log_file,
                    )

                    stale_heatmap_path = save_stale_count_heatmap_grid(
                        save_dir=stale_heatmap_dir,
                        step=t,
                        llm_call=llm_call_count,
                        fixed_image=fixed_img,
                        wrist_image=wrist_img,
                        patch_stale_count=patch_stale_count,
                        fixed_token_start=fixed_token_start,
                        wrist_token_start=wrist_token_start,
                        num_patches_per_image=num_patches_per_image,
                    )
                    log_message(
                        f"[STALE HEATMAP] step={t:03d} llm_call={llm_call_count} path={stale_heatmap_path}",
                        log_file,
                    )

                # ── exact reuse / recompute stats ─────────────────────────
                exact_fixed_reuse_count, exact_wrist_reuse_count = _count_reusable_tokens_by_view(
                    list(exact_reused_set),
                    fixed_token_start,
                    wrist_token_start,
                    num_patches_per_image,
                )

                exact_fixed_reuse_ratio = exact_fixed_reuse_count / num_patches_per_image * 100
                exact_wrist_reuse_ratio = exact_wrist_reuse_count / num_patches_per_image * 100
                exact_visual_reuse_ratio = exact_reuse_count / (num_patches_per_image * 2) * 100

                recomputed_candidate_count = max(0, reusable_candidate_count - exact_reuse_count)
                recomputed_candidate_ratio = (
                    recomputed_candidate_count / reusable_candidate_count * 100
                    if reusable_candidate_count > 0 else 0.0
                )

                # ── actual pruning ratio from exact reused patches ──────────
                # exact_reused_set = modeling_llama.py에서 실제로 reuse/prune된 visual token indices
                actual_fixed_pruned = float(exact_fixed_reuse_count)
                actual_wrist_pruned = float(exact_wrist_reuse_count)
                actual_total_visual_pruned = actual_fixed_pruned + actual_wrist_pruned

                actual_fixed_pruned_ratio = actual_fixed_pruned / num_patches_per_image * 100
                actual_wrist_pruned_ratio = actual_wrist_pruned / num_patches_per_image * 100
                actual_total_pruned_ratio = actual_total_visual_pruned / (num_patches_per_image * 2) * 100

                # q_ratios는 디버그/참고용으로만 따로 저장
                q_ratios = getattr(llama_model, "last_q_ratios", {}) or {}
                q_total = q_ratios.get("total", {}) if isinstance(q_ratios, dict) else {}

                q_total_pruned_ratio = float(q_total.get("pruned", 0.0)) if q_total else 0.0
                q_total_remaining_ratio = float(q_total.get("remaining", 0.0)) if q_total else 0.0

                # ── wrist camera pose/motion stats per LLM call ────────────
                if T_w_c_curr is not None:
                    cam_pos = np.asarray(T_w_c_curr[:3, 3], dtype=np.float32)
                else:
                    cam_pos = None

                if prev_signal_T_w_c is not None and T_w_c_curr is not None:
                    prev_cam_pos = np.asarray(prev_signal_T_w_c[:3, 3], dtype=np.float32)
                    cam_trans_delta = float(np.linalg.norm(cam_pos - prev_cam_pos))
                    cam_rot_delta_rad = _rotation_delta_angle_rad(prev_signal_T_w_c, T_w_c_curr)
                else:
                    cam_trans_delta = np.nan
                    cam_rot_delta_rad = np.nan

                log_message("=" * 20 + " SIGNAL " + "=" * 20, log_file)
                log_message(
                    f"[CALL SIGNAL] step={t:03d} llm_call={llm_call_count} "
                    f"hard_refresh={hard_refresh_step} "
                    f"hole_ratio={hole_ratio if hole_ratio is not None else 'NA'} "
                    f"cam_trans_delta={cam_trans_delta:.6f} "
                    f"cam_rot_delta_rad={cam_rot_delta_rad:.6f} "
                    f"fixed_critical_count={len(fixed_task_relevant_indices)} fixed_critical_ratio={fixed_critical_ratio:.2f}% "
                    f"wrist_critical_count={len(wrist_task_relevant_indices)} wrist_critical_ratio={wrist_critical_ratio:.2f}% "
                    f"reusable_candidates={reusable_candidate_count} "
                    f"exact_reused={exact_reuse_count} "
                    f"exact_reuse_ratio_of_candidates={exact_reuse_ratio_of_candidates:.2f}% "
                    f"exact_fixed_reuse_ratio={exact_fixed_reuse_ratio:.2f}% "
                    f"exact_wrist_reuse_ratio={exact_wrist_reuse_ratio:.2f}% "
                    f"recomputed_candidate_ratio={recomputed_candidate_ratio:.2f}% "
                    f"actual_fixed_pruned_ratio={actual_fixed_pruned_ratio:.2f}% "
                    f"actual_wrist_pruned_ratio={actual_wrist_pruned_ratio:.2f}% "
                    f"actual_total_visual_pruned_ratio={actual_total_pruned_ratio:.2f}% "
                    f"q_total_pruned_ratio={q_total_pruned_ratio:.2f}% "
                    # f"fixed_cache_age_mean={fixed_cache_age_mean:.2f} fixed_cache_age_max={fixed_cache_age_max:.0f} "
                    # f"wrist_cache_age_mean={wrist_cache_age_mean:.2f} wrist_cache_age_max={wrist_cache_age_max:.0f} "
                    f"fixed_recency_blocked_count={fixed_recency_blocked_count} "
                    f"wrist_recency_blocked_count={wrist_recency_blocked_count} "
                    f"fixed_recency_blocked_ratio={fixed_recency_blocked_ratio:.2f}% "
                    f"wrist_recency_blocked_ratio={wrist_recency_blocked_ratio:.2f}% "
                    f"fixed_stale_ge_threshold_count={fixed_stale_ge_threshold_count} "
                    f"fixed_eq{cfg.stale_force_threshold}_index_count={fixed_stale_eq_threshold_count} "
                    f"wrist_eq{cfg.stale_force_threshold}_index_count={wrist_stale_eq_threshold_count} "
                    f"total_eq{cfg.stale_force_threshold}_index_count={total_stale_eq_threshold_count} "
                    # f"fixed_stale_ge_threshold_ratio={fixed_stale_ge_threshold_ratio:.2f}% "
                    f"wrist_stale_ge_threshold_count={wrist_stale_ge_threshold_count} "
                    # f"wrist_stale_ge_threshold_ratio={wrist_stale_ge_threshold_ratio:.2f}% "
                    f"fixed_stale_reuse_ratio={fixed_stale_reuse_ratio:.2f}% "
                    f"wrist_stale_reuse_ratio={wrist_stale_reuse_ratio:.2f}% "
                    f"fixed_stale_critical_overlap={fixed_stale_critical_overlap_count} "
                    f"fixed_stale_critical_overlap_ratio={fixed_stale_critical_overlap_ratio:.2f}% "
                    f"wrist_stale_critical_overlap={wrist_stale_critical_overlap_count} "
                    f"wrist_stale_critical_overlap_ratio={wrist_stale_critical_overlap_ratio:.2f}% "
                    f"cam_pos={cam_pos.tolist() if cam_pos is not None else 'NA'}",
                    log_file,
                )
                log_message("=" * 48, log_file)
                # ──────────────────────────────────────────────────────────
                # ------------------------------------------------------------
                # Update previous wrist camera pose for the NEXT LLM call.
                # 여기서 업데이트해야 다음 LLM call에서 현재 call pose를 previous pose로 사용함.
                # ------------------------------------------------------------
                if cfg.use_dynam_cache and will_call_llm and T_w_c_curr is not None:
                    prev_signal_T_w_c = np.asarray(T_w_c_curr, dtype=np.float32).copy()         

                # # ── VISUALIZE REUSE PATCH ─────────────────────────

                # -----------------------------------------------------------------
                # Rolling-anchor update
                # -----------------------------------------------------------------
                if cfg.use_dynam_cache and cfg.use_rolling_anchor:
                    # After each LLM forward, the current frame becomes the source
                    # for the next LLM call. This keeps cache source and warp source aligned.
                    cache_anchor_T_w_c = T_w_c_curr
                    cache_anchor_img = wrist_img
                    cache_anchor_cache = clone_cache(last_caches)

                    if llm_call_count <= 5 or (llm_call_count % 10 == 0):
                        log_message(
                            f"[ROLLING ANCHOR] step={t:03d} "
                            f"llm_call={llm_call_count} "
                            f"updated=True "
                            f"cache_anchor_cache_none={cache_anchor_cache is None}",
                            log_file,
                        )

                if progressive_debug:
                    for item in progressive_debug:
                        if item["type"] == "prune_after":
                            log_message(
                                f"[DBG q_len] layer={item['layer']:02d} "
                                f"q_after={item['q_after']} "
                                f"reused_so_far={item['reused_so_far']} "
                                f"reuse_ratio_actual={item['reuse_ratio_actual']:.2f}% "
                                f"reuse_ratio_of_candidates={item['reuse_ratio_of_candidates']:.2f}%",
                                log_file,
                            )

                        elif item["type"] == "summary":
                            log_message(
                                f"[DBG summary] original_q_len={item['original_q_len']} "
                                f"final_q_len={item['final_q_len']} "
                                f"total_reusable_candidates={item['total_reusable_candidates']} "
                                f"total_reused={item['total_reused']} "
                                f"final_compute_ratio={item['final_compute_ratio']:.2f}% "
                                f"reuse_ratio_of_all_tokens={item['reuse_ratio_of_all_tokens']:.2f}% "
                                f"reuse_ratio_of_candidates={item['reuse_ratio_of_candidates']:.2f}%",
                                log_file,
                            )

                if cfg.use_dynam_cache and force_keyframe_refresh and not cfg.use_rolling_anchor:
                    keyframe_T_w_c = T_w_c_curr
                    keyframe_img = wrist_img

                    if cfg.use_recency_cap:
                        # 이전 recencycap8 방식:
                        # keyframe_cache는 유지하고, recency cap이 stale patch recompute를 관리
                        pass
                    else:
                        # recency cap이 없을 때만 기존 full keyframe cache 갱신
                        keyframe_cache = clone_cache(last_caches)
                        static_indices = all_visual_token_indices
                        warped_caches = None
                        patch_stale_count.clear()

                    log_message(
                        f"[KEYFRAME] refreshed at llm_call={llm_call_count}, "
                        f"step={t:03d}, interval={cfg.keyframe_interval}, "
                        f"use_recency_cap={cfg.use_recency_cap}",
                        log_file,
                    )
            else:
                _t_model_start = time.perf_counter()
                _t_model_end = _t_model_start

            _t_post_model = time.perf_counter()

            

            # -----------------------------------------------------------------
            # [Step 5] Keyframe 갱신
            # -----------------------------------------------------------------
            if cfg.use_dynam_cache and is_keyframe_step:
                # warping anchor는 항상 갱신
                keyframe_T_w_c   = T_w_c_curr
                keyframe_img     = wrist_img
                # 초기 keyframe_cache는 use_recency_cap 무관하게 항상 초기화 필요
                # (episode 첫 step이라 기존 cache가 없음)
                keyframe_cache   = clone_cache(last_caches)
                static_indices   = all_visual_token_indices

                if not cfg.use_recency_cap:
                    patch_stale_count.clear()

                log_message("초기 키프레임 캐시 갱신됨", log_file)

            # -----------------------------------------------------------------
            # [Step 6] 액션 실행 + control frequency 로깅
            # -----------------------------------------------------------------
                
            # Get action from queue & Process action
            action = process_action(action_queue.popleft(), cfg.model_family)

            # # ── ACTION LOG (별도 깨끗한 로그) ──────────────────────────
            # _action_list = action.tolist() if hasattr(action, "tolist") else list(action)
            # if action_log_file:
            #     action_log_file.write(
            #         f"[ACTION] step={t:03d} "
            #         f"phase={current_grasp_phase} "
            #         f"grasp_risk={current_grasp_risk} "
            #         f"action={[f'{v:.4f}' for v in _action_list]} "
            #         f"eef=({float(obs['robot0_eef_pos'][0]):.4f},{float(obs['robot0_eef_pos'][1]):.4f},{float(obs['robot0_eef_pos'][2]):.4f}) "
            #         f"gripper_qpos=({float(obs['robot0_gripper_qpos'][0]):.4f},{float(obs['robot0_gripper_qpos'][1]):.4f})\n"
            #     )
            #     action_log_file.flush()
            # # ────────────────────────────────────────────────────────────


            # Execute action in environment
            _t_env_start = time.perf_counter()
            obs, reward, done, info = env.step(action.tolist())
            _t_env_end = time.perf_counter()

            # ── STEP SIGNAL 로그 ────────────────────────────────────────
            new_eef = obs["robot0_eef_pos"]
            new_gripper = obs["robot0_gripper_qpos"]

            action_np = np.asarray(action, dtype=np.float32).reshape(-1)

            # action command components
            action_dx = float(action_np[0]) if action_np.size > 0 else 0.0
            action_dy = float(action_np[1]) if action_np.size > 1 else 0.0
            action_dz = float(action_np[2]) if action_np.size > 2 else 0.0
            action_droll = float(action_np[3]) if action_np.size > 3 else 0.0
            action_dpitch = float(action_np[4]) if action_np.size > 4 else 0.0
            action_dyaw = float(action_np[5]) if action_np.size > 5 else 0.0
            action_gripper = float(action_np[-1]) if action_np.size > 0 else 0.0

            action_translation_norm = float(np.linalg.norm(action_np[:3])) if action_np.size >= 3 else 0.0
            action_rotation_norm = float(np.linalg.norm(action_np[3:6])) if action_np.size >= 6 else 0.0

            if prev_eef_obs is not None:
                dx = float(new_eef[0] - prev_eef_obs["robot0_eef_pos"][0])
                dy = float(new_eef[1] - prev_eef_obs["robot0_eef_pos"][1])
                dz = float(new_eef[2] - prev_eef_obs["robot0_eef_pos"][2])

                delta_pos = np.asarray([dx, dy, dz], dtype=np.float32)
                eef_speed = float(np.linalg.norm(delta_pos))

                d_g0 = float(new_gripper[0] - prev_eef_obs["robot0_gripper_qpos"][0])
                d_g1 = float(new_gripper[1] - prev_eef_obs["robot0_gripper_qpos"][1])

                # gripper width 변화량: 양수/음수 방향성 있음
                dgripper = d_g0 - d_g1

                # 두 finger가 같은 방향으로 움직이는 정도.
                # 정상 mirrored motion이면 0에 가까움.
                dgripper_asym = d_g0 + d_g1

                recent_eef_speeds.append(eef_speed)
                recent_dgripper_asyms.append(dgripper_asym)

                # acceleration / jerk
                if prev_step_eef_speed is None:
                    accel = 0.0
                else:
                    accel = float(eef_speed - prev_step_eef_speed)

                if prev_step_accel is None:
                    jerk = 0.0
                else:
                    jerk = float(accel - prev_step_accel)

                direction_cosine = _safe_direction_cosine(delta_pos, prev_delta_pos)

            else:
                dx = dy = dz = eef_speed = 0.0
                dgripper = dgripper_asym = 0.0
                delta_pos = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
                accel = 0.0
                jerk = 0.0
                direction_cosine = np.nan

            gripper_width = float(new_gripper[0] - new_gripper[1])
            asym_window_max = max((abs(v) for v in recent_dgripper_asyms), default=0.0)
            asym_window_std = float(np.std(recent_dgripper_asyms)) if len(recent_dgripper_asyms) > 1 else 0.0
            action_eef_ratio = float(action_translation_norm / (eef_speed + 1e-6))

            # ── transition / contact signal window ─────────────────────
            # 위치: action_eef_ratio 계산 직후, progress / oscillation stats 전에 넣기
            if prev_eef_obs is not None:
                recent_abs_accels.append(abs(float(accel)))
                recent_abs_jerks.append(abs(float(jerk)))
                recent_action_eef_ratios.append(float(action_eef_ratio))

                if len(recent_eef_speeds) > 0:
                    recent_speed_max = max(float(x) for x in recent_eef_speeds)
                    current_speed = float(eef_speed)

                    speed_drop_ratio = max(
                        0.0,
                        (recent_speed_max - current_speed) / (recent_speed_max + 1e-6),
                    )
                else:
                    speed_drop_ratio = 0.0

                recent_speed_drop_ratios.append(float(speed_drop_ratio))
            else:
                speed_drop_ratio = 0.0
            # ──────────────────────────────────────────────────────────

            # ── progress / oscillation stats ─────────────────────────
            recent_eef_positions.append(np.asarray(new_eef, dtype=np.float32).copy())

            if np.isfinite(direction_cosine):
                recent_direction_cosines.append(float(direction_cosine))

            if len(recent_eef_positions) >= 2:
                pos_arr = np.stack(list(recent_eef_positions), axis=0)
                step_deltas = np.diff(pos_arr, axis=0)

                recent_path_length = float(
                    np.sum(np.linalg.norm(step_deltas, axis=1))
                )
                recent_net_displacement = float(
                    np.linalg.norm(pos_arr[-1] - pos_arr[0])
                )
                straightness = float(
                    recent_net_displacement / (recent_path_length + 1e-8)
                )
            else:
                recent_path_length = 0.0
                recent_net_displacement = 0.0
                straightness = 0.0

            direction_dip_count = int(
                sum(1 for c in recent_direction_cosines if np.isfinite(c) and c < 0.3)
            )

            direction_cosine_std = float(np.std(recent_direction_cosines)) if len(recent_direction_cosines) > 1 else 0.0
            # ─────────────────────────────────────────────────────────

            _action_str = ",".join(f"{v:.3f}" for v in action_np.tolist())

            log_message("-" * 20 + " STEP " + "-" * 20, log_file)
            log_message(
                f"[STEP] step={t:03d} "
                f"eef_pos=({float(new_eef[0]):.4f},{float(new_eef[1]):.4f},{float(new_eef[2]):.4f}) "
                f"delta=({dx:.5f},{dy:.5f},{dz:.5f}) "
                f"eef_speed={eef_speed:.5f} "
                f"accel={accel:.6f} jerk={jerk:.6f} "
                f"direction_cosine={direction_cosine:.4f} "
                f"gripper=({float(new_gripper[0]):.4f},{float(new_gripper[1]):.4f}) "
                f"gripper_width={gripper_width:.6f} "
                f"dgripper={dgripper:.6f} dgripper_asym={dgripper_asym:.6f} "
                f"asym_win_max={asym_window_max:.6f} asym_win_std={asym_window_std:.6f} "
                f"action_trans_norm={action_translation_norm:.6f} "
                f"action_rot_norm={action_rotation_norm:.6f} "
                f"action_eef_ratio={action_eef_ratio:.3f} "
                f"recent_net_disp={recent_net_displacement:.6f} "
                f"recent_path_len={recent_path_length:.6f} "
                f"straightness={straightness:.3f} "
                f"direction_dip_count={direction_dip_count} "
                f"direction_cosine_std={direction_cosine_std:.3f} "
                f"action=[{_action_str}]",
                log_file,
            )
            log_message("-" * 46, log_file)

            step_signal_records.append({
                "step": t,

                # raw EEF position
                "eef_x": float(new_eef[0]),
                "eef_y": float(new_eef[1]),
                "eef_z": float(new_eef[2]),

                # delta motion
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "eef_speed": eef_speed,
                "accel": accel,
                "abs_accel": abs(accel),
                "jerk": jerk,
                "abs_jerk": abs(jerk),
                "direction_cosine": direction_cosine,

                # gripper
                "gripper_0": float(new_gripper[0]),
                "gripper_1": float(new_gripper[1]),
                "gripper_width": gripper_width,
                "dgripper": dgripper,
                "dgripper_asym": dgripper_asym,
                "abs_dgripper_asym": abs(dgripper_asym),
                "asym_window_max": asym_window_max,
                "asym_window_std": asym_window_std,

                # action command
                "action_dx": action_dx,
                "action_dy": action_dy,
                "action_dz": action_dz,
                "action_droll": action_droll,
                "action_dpitch": action_dpitch,
                "action_dyaw": action_dyaw,
                "action_gripper": action_gripper,
                "action_translation_norm": action_translation_norm,
                "action_rotation_norm": action_rotation_norm,
                "action_eef_ratio": action_eef_ratio,
                # progress / oscillation
                "recent_net_displacement": recent_net_displacement,
                "recent_path_length": recent_path_length,
                "straightness": straightness,
                "direction_dip_count": direction_dip_count,
                "direction_cosine_std": direction_cosine_std,
            })

            prev_eef_obs = {
                "robot0_eef_pos": new_eef.copy(),
                "robot0_gripper_qpos": new_gripper.copy(),
            }

            prev_delta_pos = delta_pos.copy()
            prev_step_eef_speed = eef_speed
            prev_step_accel = accel
            # ──────────────────────────────────────────────────────────
            # ──────────────────────────────────────────────────────────
            # if action_log_file:
            #     new_eef = obs['robot0_eef_pos']
            #     new_gripper = obs['robot0_gripper_qpos']
            #     if prev_eef_obs is not None:
            #         dx = float(new_eef[0] - prev_eef_obs['robot0_eef_pos'][0])
            #         dy = float(new_eef[1] - prev_eef_obs['robot0_eef_pos'][1])
            #         dz = float(new_eef[2] - prev_eef_obs['robot0_eef_pos'][2])
            #         eef_speed = (dx**2 + dy**2 + dz**2) ** 0.5

            #         # 각 finger의 raw signed delta (abs+abs 합산 방식 폐기)
            #         d_g0 = float(new_gripper[0] - prev_eef_obs['robot0_gripper_qpos'][0])
            #         d_g1 = float(new_gripper[1] - prev_eef_obs['robot0_gripper_qpos'][1])

            #         # width(=벌어진 정도) 변화량: 음수=닫힘, 양수=열림
            #         # g0>0, g1<0 정상 배치 기준이며 abs() 없이 직접 차분이라
            #         # 부호 역전 구간에서도 안전하게 동작
            #         dgripper = d_g0 - d_g1

            #         # 두 finger가 "같은 방향"으로 움직인 정도.
            #         # 정상적인 mirrored 닫힘/열림이면 0에 가깝고,
            #         # 0이 아니면 비대칭 움직임(전조 신호 후보)
                                        
            #         recent_eef_speeds.append(float(eef_speed))
            #         recent_dgrippers.append(float(dgripper))
            #         recent_gripper_widths.append(float(new_gripper[0] - new_gripper[1]))
            #     else:
            #         dx, dy, dz, eef_speed = 0.0, 0.0, 0.0, 0.0
            #         d_g0, d_g1, dgripper, dgripper_asym = 0.0, 0.0, 0.0, 0.0

            #     action_log_file.write(
            #         f"[ACTION RESULT] step={t:03d} "
            #         f"phase={current_grasp_phase} "
            #         f"grasp_risk={current_grasp_risk} "
            #         f"new_eef=({float(new_eef[0]):.4f},{float(new_eef[1]):.4f},{float(new_eef[2]):.4f}) "
            #         f"dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} eef_speed={eef_speed:.4f} "
            #         f"gripper=({float(new_gripper[0]):.4f},{float(new_gripper[1]):.4f}) "
            #         f"dgripper={dgripper:.4f} d_g0={d_g0:.4f} d_g1={d_g1:.4f}\n"
            #     )
            #     action_log_file.flush()

            # prev_eef_obs = {
            #     'robot0_eef_pos': obs['robot0_eef_pos'].copy(),
            #     'robot0_gripper_qpos': obs['robot0_gripper_qpos'].copy(),
            # }
            # # ────────────────────────────────────────────────────────────
            
            # # ---------------------------------- [PHASE 구간] -----------------------------------------------
            # curr_gripper = float(abs(obs['robot0_gripper_qpos'][0]) + abs(obs['robot0_gripper_qpos'][1]))

            # if prev_eef_obs is not None:
            #     prev_gripper = float(abs(prev_eef_obs['robot0_gripper_qpos'][0]) + abs(prev_eef_obs['robot0_gripper_qpos'][1]))
            #     dx = float(obs['robot0_eef_pos'][0] - prev_eef_obs['robot0_eef_pos'][0])
            #     dy = float(obs['robot0_eef_pos'][1] - prev_eef_obs['robot0_eef_pos'][1])
            #     dz = float(obs['robot0_eef_pos'][2] - prev_eef_obs['robot0_eef_pos'][2])
            #     dgripper = float(curr_gripper - prev_gripper)
            # else:
            #     prev_gripper = curr_gripper
            #     dx, dy, dz, dgripper = 0.0, 0.0, 0.0, 0.0

            # # baseline 업데이트
            # if gripper_baseline is None:
            #     result = update_gripper_baseline(gripper_history, curr_gripper)
            #     if result is not None:
            #         gripper_baseline = result

            # # phase 감지
            # phase = detect_phase(
            #     curr_gripper, prev_gripper,
            #     obs['robot0_eef_pos'], prev_eef_obs['robot0_eef_pos'] if prev_eef_obs else obs['robot0_eef_pos'],
            #     gripper_baseline,
            # )

            # last_phase = phase


            # eef_log.append({
            #     'step': t, 'x': float(obs['robot0_eef_pos'][0]),
            #     'y': float(obs['robot0_eef_pos'][1]), 'z': float(obs['robot0_eef_pos'][2]),
            #     'gripper': curr_gripper, 'dx': dx, 'dy': dy, 'dz': dz,
            #     'dgripper': dgripper, 'phase': phase,
            # })

            # log_message(
            #     f"[EEF] step={t:03d} phase={phase} "
            #     f"x={float(obs['robot0_eef_pos'][0]):.4f} y={float(obs['robot0_eef_pos'][1]):.4f} "
            #     f"z={float(obs['robot0_eef_pos'][2]):.4f} gripper={curr_gripper:.4f} "
            #     f"dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} dgripper={dgripper:.4f}",
            #     log_file
            # )

            # prev_eef_obs = {
            #     'robot0_eef_pos': obs['robot0_eef_pos'].copy(),
            #     'robot0_gripper_qpos': obs['robot0_gripper_qpos'].copy(),
            # }

            # # ---------------------------- [PHASE 구간] ---------------------------------------------------------

            current_chunk_env_s += (_t_env_end - _t_env_start)
            current_chunk_actions_executed += 1

            control_step_time = time.perf_counter() - step_start_time
            current_hz        = 1.0 / control_step_time if control_step_time > 0 else 0.0

            _llm_cuda_ms = getattr(
                getattr(getattr(model, "language_model", None), "model", None),
                "last_cuda_time",
                None,
            )
            _p_img     = (_t_step1      - step_start_time) * 1000
            _p_model   = (_t_model_end  - _t_model_start)  * 1000
            _p_postmdl = (_t_post_model - _t_model_end)    * 1000
            _p_env     = (_t_env_end    - _t_env_start)    * 1000
            _p_total   = control_step_time * 1000
            _p_other   = _p_total - (_p_img + _p_model + _p_postmdl + _p_env)

            # if will_call_llm:
            #     profile_msg = (
            #         f"[PROFILE t={t:03d}] "
            #         f"dynam_cache={'Y' if cfg.use_dynam_cache else 'N'} "
            #         f"kv_reuse={'N' if cfg.disable_kv_cache_reuse else 'Y'}\n"
            #         f"  img_prep  : {_p_img:7.1f} ms\n"                          # ← 추가
            #         f"  model_fwd : {_p_model:7.1f} ms  (get_action / LLM)"
            #         + (f"  [LLM CUDA={_llm_cuda_ms:.1f}ms]" if _llm_cuda_ms else "") + "\n"
            #         f"  post_mdl  : {_p_postmdl:7.1f} ms\n"
            #         f"  env_step  : {_p_env:7.1f} ms\n"
            #         f"  other     : {_p_other:7.1f} ms\n"
            #         f"  TOTAL     : {_p_total:7.1f} ms -> {1000 / max(_p_total, 1e-9):.2f} Hz"
            #     )
            #     log_message(profile_msg, log_file)
 
            perf_counters["total_steps"] += 1
            perf_counters["total_time"]  += control_step_time
            perf_counters["total_wall_ms"] += _p_total
            perf_counters["total_all_steps"] += 1
            perf_counters["total_reusable_fixed_tokens"] += fixed_reusable_count
            perf_counters["total_reusable_wrist_tokens"] += wrist_reusable_count

            episode_total_wall_ms += _p_total
            episode_all_steps += 1

            avg_step_ms = episode_total_wall_ms / episode_all_steps
            avg_ctrl_hz = 1000.0 / avg_step_ms
 
            queue_remaining = len(action_queue)

            # To calculate control frequency
            if current_chunk_size > 0 and current_chunk_actions_executed == current_chunk_size:
                chunk_wall_end = time.perf_counter()
                current_real_control_time_s = (
                    chunk_wall_end - chunk_wall_start
                    if chunk_wall_start is not None
                    else current_chunk_inference_s + current_chunk_env_s  # fallback
                )
                current_real_control_hz = (
                    current_chunk_size / current_real_control_time_s
                    if current_real_control_time_s > 0 else 0.0
                )

                real_control_total_time_s += current_real_control_time_s
                real_control_total_actions += current_chunk_size
                perf_counters["real_control_total_time_s"] += current_real_control_time_s
                perf_counters["real_control_total_actions"] += current_chunk_size

                average_real_control_hz = (
                    real_control_total_actions / real_control_total_time_s
                    if real_control_total_time_s > 0 else 0.0
                )

                log_message(
                    f"[Real Control Frequency] step={t:03d} "
                    f"current_real_control_hz={current_real_control_hz:.4f} Hz | "
                    f"average_real_control_hz={average_real_control_hz:.4f} Hz | "
                    f"chunk_size={current_chunk_size} | "
                    f"inference_time={current_chunk_inference_s * 1000:.1f} ms | "
                    f"env_time_sum={current_chunk_env_s * 1000:.1f} ms | "
                    f"chunk_wall_time={current_real_control_time_s * 1000:.1f} ms",  # 이름 변경
                    log_file,
                )
                current_chunk_size = 0
                current_chunk_actions_executed = 0
                current_chunk_inference_s = 0.0
                current_chunk_env_s = 0.0
                chunk_wall_start = None    # ← 리셋
                
            if will_call_llm:

                current_fixed_reuse_ratio = (
                    fixed_reusable_count / num_patches_per_image * 100
                    if will_call_llm and not hard_refresh_step
                    else 0.0
                )
                current_wrist_reuse_ratio = (
                    wrist_reusable_count / num_patches_per_image * 100
                    if will_call_llm and not hard_refresh_step
                    else 0.0
                )

                if log_file:
                    log_file.write(
                        f"[Control] step={t:03d} | "
                        f"step_time_s={control_step_time:.6f} | "
                        f"current_hz={current_hz:.4f} Hz | "
                        f"reusable_patches={reusable_count} | "
                        f"queue_remaining={queue_remaining}\n"
                    )
                    log_file.flush()
                log_message(
                    f"Current Control Frequency: {current_hz:.2f} Hz "
                    f"(this step: {_p_total:.1f} ms) | "
                    f"Average Control Frequency: {avg_ctrl_hz:.2f} Hz "
                    f"(avg step: {avg_step_ms:.1f} ms) | "
                    f"Token Reusing Ratio (Fixed): {current_fixed_reuse_ratio:.2f} % | "
                    f"Token Reusing Ratio (Wrist): {current_wrist_reuse_ratio:.2f} %",
                    log_file,
                )

                call_signal_records.append({
                    "step": t,
                    "hole_ratio": hole_ratio,

                    "fixed_critical_ratio": fixed_critical_ratio,
                    "wrist_critical_ratio": wrist_critical_ratio,
                    "fixed_reuse_ratio": current_fixed_reuse_ratio,
                    "wrist_reuse_ratio": current_wrist_reuse_ratio,

                    # exact reuse
                    "exact_reuse_ratio_of_candidates": exact_reuse_ratio_of_candidates,
                    "exact_fixed_reuse_ratio": exact_fixed_reuse_ratio,
                    "exact_wrist_reuse_ratio": exact_wrist_reuse_ratio,
                    "exact_visual_reuse_ratio": exact_visual_reuse_ratio,
                    "recomputed_candidate_ratio": recomputed_candidate_ratio,

                    # actual pruning ratio from LLaMA
                    "actual_fixed_pruned_ratio": actual_fixed_pruned_ratio,
                    "actual_wrist_pruned_ratio": actual_wrist_pruned_ratio,
                    "actual_total_pruned_ratio": actual_total_pruned_ratio,

                    # optional q-ratio debug from llama_model.last_q_ratios
                    "q_total_pruned_ratio": q_total_pruned_ratio,
                    "q_total_remaining_ratio": q_total_remaining_ratio,

                    # stale == threshold index count
                    "fixed_stale_eq_threshold_count": fixed_stale_eq_threshold_count,
                    "wrist_stale_eq_threshold_count": wrist_stale_eq_threshold_count,
                    "total_stale_eq_threshold_count": total_stale_eq_threshold_count,

                    # shorter aliases for visualization/debug
                    "fixed_eq8_index_count": fixed_stale_eq_threshold_count,
                    "wrist_eq8_index_count": wrist_stale_eq_threshold_count,
                    "total_eq8_index_count": total_stale_eq_threshold_count,

                    # recency-cap blocked candidates before current forward
                    "fixed_recency_blocked_count": fixed_recency_blocked_count,
                    "wrist_recency_blocked_count": wrist_recency_blocked_count,
                    "fixed_recency_blocked_ratio": fixed_recency_blocked_ratio,
                    "wrist_recency_blocked_ratio": wrist_recency_blocked_ratio,

                    # stale reuse / critical overlap
                    "fixed_stale_reuse_ratio": fixed_stale_reuse_ratio,
                    "wrist_stale_reuse_ratio": wrist_stale_reuse_ratio,
                    "fixed_stale_critical_overlap_ratio": fixed_stale_critical_overlap_ratio,
                    "wrist_stale_critical_overlap_ratio": wrist_stale_critical_overlap_ratio,
                    "fixed_stale_critical_overlap_count": fixed_stale_critical_overlap_count,
                    "wrist_stale_critical_overlap_count": wrist_stale_critical_overlap_count,

                    # camera motion
                    "cam_trans_delta": cam_trans_delta,
                    "cam_rot_delta_rad": cam_rot_delta_rad,

                    # adaptive pruning phase
                    "global_adaptive_phase": (
                        adaptive_phase_info.get("global_phase", "NA")
                        if adaptive_phase_info is not None else "NA"
                    ),
                    "adaptive_risk_score": (
                        adaptive_phase_info.get("risk_score", np.nan)
                        if adaptive_phase_info is not None else np.nan
                    ),
                    "adaptive_reasons": (
                        ",".join(adaptive_phase_info.get("reasons", []))
                        if adaptive_phase_info is not None else ""
                    ),

                    # compatibility fields: 기존 plot/debug 코드가 fixed/wrist phase key를 기대할 수 있어서 유지
                    "fixed_adaptive_phase": (
                        adaptive_phase_info.get("global_phase", "NA")
                        if adaptive_phase_info is not None else "NA"
                    ),
                    "wrist_adaptive_phase": (
                        adaptive_phase_info.get("global_phase", "NA")
                        if adaptive_phase_info is not None else "NA"
                    ),
                    "fixed_adaptive_reasons": (
                        ",".join(adaptive_phase_info.get("fixed_reasons", []))
                        if adaptive_phase_info is not None else ""
                    ),
                    "wrist_adaptive_reasons": (
                        ",".join(adaptive_phase_info.get("wrist_reasons", []))
                        if adaptive_phase_info is not None else ""
                    ),
                    "current_hz": current_hz,

                })



            if done:
                success = True
                break
            t += 1

    except Exception as e:
        import traceback
        log_message(f"Episode error: {e}", log_file)
        log_message(traceback.format_exc(), log_file)
        raise

    # # -------------------- [PHASE 구간] --------------------------------
    # save_eef_trajectory_plot(
    #     save_dir=os.path.join(cfg.video_save_dir, "eef_trajectory"),
    #     episode=total_episodes + 1,
    #     eef_log=eef_log,
    # )
    # # -------------------- [PHASE 구간] --------------------------------
    #  # ── ACTION LOG: episode 종료 시 success 기록 + 파일 닫기 ──────────
    # if action_log_file:
    #     action_log_file.write("=" * 60 + "\n")
    #     action_log_file.write(f"Success: {success}\n")
    #     action_log_file.close()
    # # ────────────────────────────────────────────────────────────────

    signal_plot_dir = os.path.join(cfg.video_save_dir, "signal_summary")
    save_episode_signal_plot(
        save_dir=signal_plot_dir,
        episode=total_episodes + 1,
        task_description=task_description,
        step_records=step_signal_records,
        call_records=call_signal_records,
        success=success,
    )

    return success, replay_images, replay_images_wrist, replay_images_heatmap, replay_images_wrist_heatmap

# =============================================================================
# Task runner
# =============================================================================
def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    perf_counters: Optional[dict] = None,
):
    if perf_counters is None:
        perf_counters = {"total_steps": 0, "total_time": 0.0}
    perf_counters.setdefault("total_wall_ms", 0.0)
    perf_counters.setdefault("total_all_steps", 0)
    perf_counters.setdefault("total_reusable_fixed_tokens", 0)
    perf_counters.setdefault("total_reusable_wrist_tokens", 0)
    perf_counters.setdefault("llm_reuse_calls", 0)                  #  실제 reuse를 적용한 LLM 호출 수
    perf_counters.setdefault("llm_reusable_fixed_tokens", 0)        #  그 LLM 호출들에서 fixed camera reusable token 총합
    perf_counters.setdefault("llm_reusable_wrist_tokens", 0)        #  그 LLM 호출들에서 wrist camera reusable token 총합
    perf_counters.setdefault("real_control_total_time_s", 0.0)
    perf_counters.setdefault("real_control_total_actions", 0)
    perf_counters.setdefault("layer_pruned_tokens", {})
    perf_counters.setdefault("layer_original_tokens", {})
    perf_counters.setdefault("layer_fixed_pruned", {})
    perf_counters.setdefault("layer_wrist_pruned", {})
    perf_counters.setdefault("layer_calls", {})
    perf_counters.setdefault("total_pruning_calls", 0)
    perf_counters.setdefault("total_pruned_tokens_sum", 0)
    perf_counters.setdefault("total_original_tokens_sum", 0)
    perf_counters.setdefault("total_fixed_pruned_sum", 0)
    perf_counters.setdefault("total_wrist_pruned_sum", 0)


    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, replay_images_wrist, replay_images_heatmap, replay_images_wrist_heatmap = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            total_episodes=total_episodes, # 왜 totla_episodes지
            perf_counters=perf_counters,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video with patch overlay already composited on top.
        save_rollout_video(
            replay_images_heatmap,
            total_episodes,
            success=success,
            task_description=f"primary overlay {task_description}",
            log_file=log_file,
            save_dir=cfg.video_save_dir,
        )
        save_rollout_video(
            replay_images_wrist_heatmap,
            total_episodes,
            success=success,
            task_description=f"wrist overlay {task_description}",
            log_file=log_file,
            save_dir=cfg.video_save_dir,
        )

        if perf_counters["total_all_steps"] > 0:
            avg_step_ms = perf_counters["total_wall_ms"] / perf_counters["total_all_steps"]
            avg_ctrl_hz = 1000.0 / avg_step_ms

            if perf_counters["llm_reuse_calls"] > 0:
                fixed_ratio = (
                    perf_counters["llm_reusable_fixed_tokens"]
                    / perf_counters["llm_reuse_calls"]
                    / 256
                    * 100
                )
                wrist_ratio = (
                    perf_counters["llm_reusable_wrist_tokens"]
                    / perf_counters["llm_reuse_calls"]
                    / 256
                    * 100
                )
            else:
                fixed_ratio = 0.0
                wrist_ratio = 0.0

            avg_real_ctrl_hz = (
                perf_counters["real_control_total_actions"] / perf_counters["real_control_total_time_s"]
                if perf_counters["real_control_total_time_s"] > 0 else 0.0
            )
            log_message(
                f"[Step Wall Hz] {avg_ctrl_hz:.2f} Hz (avg {avg_step_ms:.1f} ms, {perf_counters['total_all_steps']} steps) | "
                f"[Real Control Hz] {avg_real_ctrl_hz:.2f} Hz "
                f"({perf_counters['real_control_total_actions']} actions) | "
                f"Average LLM Reuse Ratio (Fixed): {fixed_ratio:.2f} % | "
                f"Average LLM Reuse Ratio (Wrist): {wrist_ratio:.2f} %",
                log_file,
            )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes

# =============================================================================
# Entry point
# =============================================================================

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    perf_counters = {
        "total_steps": 0,
        "total_time": 0.0,
        "total_wall_ms": 0.0,
        "total_all_steps": 0,
        "total_reusable_fixed_tokens": 0,
        "total_reusable_wrist_tokens": 0,

        # LLM 호출 기준 reuse 평균용
        "llm_reuse_calls": 0,
        "llm_reusable_fixed_tokens": 0,
        "llm_reusable_wrist_tokens": 0,

        # 실제 chunk control Hz 평균용
        "real_control_total_time_s": 0.0,
        "real_control_total_actions": 0,

        # layer별 누적 pruning 통계
        "layer_pruned_tokens": {},    # {layer_idx: 누적 pruned tokens}
        "layer_original_tokens": {},  # {layer_idx: 누적 original seq len}
        "layer_fixed_pruned": {},     # {layer_idx: 누적 fixed pruned}
        "layer_wrist_pruned": {},     # {layer_idx: 누적 wrist pruned}
        "layer_calls": {},            # {layer_idx: 호출 횟수}
        "total_pruning_calls": 0,
        "total_pruned_tokens_sum": 0,
        "total_original_tokens_sum": 0,
        "total_fixed_pruned_sum": 0,
        "total_wrist_pruned_sum": 0,
    }

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            perf_counters=perf_counters,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    
    log_message("=" * 60, log_file)
    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    if perf_counters["real_control_total_time_s"] > 0:
        avg_real_control_hz = (
            perf_counters["real_control_total_actions"]
            / perf_counters["real_control_total_time_s"]
        )
        log_message(
            f"  Avg real control Hz : {avg_real_control_hz:.4f} Hz "
            f"(over {perf_counters['real_control_total_actions']} actions)",
            log_file,
        )

    if perf_counters["llm_reuse_calls"] > 0:
        fixed_ratio = (
            perf_counters["llm_reusable_fixed_tokens"]
            / perf_counters["llm_reuse_calls"]
            / 256
            * 100
        )
        wrist_ratio = (
            perf_counters["llm_reusable_wrist_tokens"]
            / perf_counters["llm_reuse_calls"]
            / 256
            * 100
        )
    else:
        fixed_ratio = 0.0
        wrist_ratio = 0.0

    log_message(
        f"  Average LLM Reuse Ratio (Fixed): {fixed_ratio:.2f} % | "
        f"Average LLM Reuse Ratio (Wrist): {wrist_ratio:.2f} % "
        f"(over {perf_counters['llm_reuse_calls']} reuse LLM calls)",
        log_file,
    )

    if perf_counters.get("layer_calls"):
        log_message("[Final Pruning Stats]", log_file)
        for layer_idx in sorted(perf_counters["layer_calls"].keys()):
            n = perf_counters["layer_calls"][layer_idx]
            avg_p = perf_counters["layer_pruned_tokens"][layer_idx] / n
            avg_o = perf_counters["layer_original_tokens"][layer_idx] / n
            avg_f = perf_counters["layer_fixed_pruned"][layer_idx] / n
            avg_w = perf_counters["layer_wrist_pruned"][layer_idx] / n
            log_message(
                f"  L{layer_idx}: avg_pruned={avg_p:.1f}/{avg_o:.0f} ({avg_p/avg_o*100:.1f}%) | "
                f"fixed={avg_f:.1f}/256 ({avg_f/256*100:.1f}%) | "
                f"wrist={avg_w:.1f}/256 ({avg_w/256*100:.1f}%)",
                log_file,
            )
        calls = perf_counters["total_pruning_calls"]
        if calls > 0:
            avg_tp = perf_counters["total_pruned_tokens_sum"] / calls
            avg_to = perf_counters["total_original_tokens_sum"] / calls
            avg_tf = perf_counters["total_fixed_pruned_sum"] / calls
            avg_tw = perf_counters["total_wrist_pruned_sum"] / calls
            log_message(
                f"  Total: avg_pruned={avg_tp:.1f}/{avg_to:.0f} ({avg_tp/avg_to*100:.1f}%) | "
                f"fixed={avg_tf:.1f}/256 ({avg_tf/256*100:.1f}%) | "
                f"wrist={avg_tw:.1f}/256 ({avg_tw/256*100:.1f}%) "
                f"(over {calls} pruning calls)",
                log_file,
            )

    log_message("=" * 60, log_file)

    save_run_summary_plot(cfg.video_save_dir, run_id, perf_counters)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
