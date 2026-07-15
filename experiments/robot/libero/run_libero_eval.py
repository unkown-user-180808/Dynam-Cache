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

from experiments.robot.libero.cache_utils import (
    apply_kv_cache_warp,
    cache_age_stats,
    clone_cache,
    count_tokens_by_view,
    token_indices_to_patch_indices,
)
from experiments.robot.libero.warp_utils import (
    WarpTracker,
    compute_static_patch_indices,
    get_cam_T_w_c,
    get_sim_handle,
    rotation_delta_angle_rad,
    safe_direction_cosine,
)
from experiments.robot.libero.attention_utils import (
    warp_attention_map_with_mapping,
    compute_critical_patch_indices,
    normalize_attention_layer_ids,
    order_candidates_for_pruning,
    apply_view_budget_cap,
    OnlineAdaptiveRiskController,
    build_online_global_adaptive_pruning_plan,
)

from experiments.robot.libero.visualize_utils import (
    make_patch_overlay,
    save_exact_reuse_grid,
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
    "runtime_debug_logging",
    "save_runtime_debug_images",
    "save_rollout_videos",
    "save_signal_plots",
    "profile_llm_cuda",
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
    keyframe_interval: int = 0
    # If True, use previous LLM call as the cache/warp source instead of a fixed keyframe.
    use_rolling_anchor: bool = True
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
    attention_ema_alpha: float = 0.5  # new attention weight: ema = alpha * current + (1-alpha) * previous
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
    candidate_ordering_mode: str = "normattn_global"
    # ── View-aware candidate budget cap ──────────────────────────────────────
    fixed_max_reuse_ratio: float = 1.0   # fixed candidate cap (1.0 = no cap)
    wrist_max_reuse_ratio: float = 1.0   # wrist candidate cap (1.0 = no cap)
    

    # ── Progressive layer drop ───────────────────────────────────────────────
    use_adaptive_pruning: bool = False
    progressive_pruning_layers: Optional[Tuple[int, ...]] = None
    progressive_drop_ratios: Optional[Tuple[float, ...]] = None
    
    # ── DEBUG ───────────────────────────────────────────────
    runtime_debug_logging: bool = False
    save_runtime_debug_images: bool = False
    save_rollout_videos: bool = False
    save_signal_plots: bool = False
    profile_llm_cuda: bool = False
    debug_progressive_drop: bool = False

    # ── Critical patch에 사용할 attention source ─────────────────────
    # "mixed"         : 기존 action+text 전체 query map
    # "text_only"     : prompt text 전체 query map
    # "content_words" : task content words only map
    # "status_only"   : robot proprio/status token map
    # "action_only"   : action tokens map
    critical_attention_mode: str = "content_words"

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


def _log_llama_perf_metrics(
    log_file,
    model,
    perf_counters=None,
    emit_log: bool = False,
) -> None:
    language_model = getattr(model, "language_model", None)
    llama_model = getattr(language_model, "model", None)
    if llama_model is None:
        return

    if emit_log and log_file is not None:
        current_cuda_latency = getattr(llama_model, "last_cuda_time", None)
        average_cuda_latency = getattr(llama_model, "last_cuda_time_avg", None)
        average_tflops = getattr(llama_model, "last_tflops_avg", None)
        if average_tflops is not None:
            cuda_part = (
                f"Current CUDA latency: {float(current_cuda_latency):.6f} ms | "
                f"Average CUDA latency: {float(average_cuda_latency):.6f} ms, "
                if current_cuda_latency is not None and average_cuda_latency is not None
                else ""
            )
            log_file.write(
                f"{cuda_part}Average TFLOPs: {float(average_tflops):.6f}\n"
            )

    q_ratios = getattr(llama_model, "last_q_ratios", None)
    if not q_ratios:
        return

    if emit_log and log_file is not None:
        layer_lines = []
        for layer_idx, values in sorted(
            (k, v) for k, v in q_ratios.items() if k != "total"
        ):
            layer_lines.append(
                f"L{layer_idx}: remaining={values['remaining']:.1f}% "
                f"pruned={values['pruned']:.1f}% "
                f"({values['pruned_tokens']}/{values['seq_length']})"
            )
        if layer_lines:
            log_file.write(f"Q Ratio (per layer): {' | '.join(layer_lines)}\n")

    if perf_counters is not None:
        for layer_idx, values in q_ratios.items():
            if layer_idx == "total":
                continue
            perf_counters["layer_calls"][layer_idx] = perf_counters["layer_calls"].get(layer_idx, 0) + 1
            perf_counters["layer_pruned_tokens"][layer_idx] = perf_counters["layer_pruned_tokens"].get(layer_idx, 0) + values["pruned_tokens"]
            perf_counters["layer_original_tokens"][layer_idx] = perf_counters["layer_original_tokens"].get(layer_idx, 0) + values["original_seq_length"]
            perf_counters["layer_fixed_pruned"][layer_idx] = perf_counters["layer_fixed_pruned"].get(layer_idx, 0) + values.get("fixed_pruned", 0)
            perf_counters["layer_wrist_pruned"][layer_idx] = perf_counters["layer_wrist_pruned"].get(layer_idx, 0) + values.get("wrist_pruned", 0)

        total = q_ratios.get("total")
        if total is not None:
            perf_counters["total_pruning_calls"] += 1
            perf_counters["total_pruned_tokens_sum"] += total["pruned_tokens"]
            perf_counters["total_original_tokens_sum"] += total["original_seq_length"]
            perf_counters["total_fixed_pruned_sum"] += total.get("fixed_pruned", 0)
            perf_counters["total_wrist_pruned_sum"] += total.get("wrist_pruned", 0)

_RUNTIME_DEBUG_LOGGING = False
_NOISY_RUNTIME_PREFIXES = (
    "[STEP]", "[CALL SIGNAL]", "[PREFWD", "[STALE", "[RECENCY CAP]",
    "[ADAPTIVE PRUNING", "[ADAPTIVE PHASE", "[EXACT REUSE", "[ROLLING ANCHOR]",
    "[Control]", "Current Control Frequency", "-------------------- STEP",
    "==================== SIGNAL",
)


def log_message(message: str, log_file=None, *, force: bool = False) -> None:
    stripped = message.lstrip()
    noisy = stripped.startswith(_NOISY_RUNTIME_PREFIXES)
    if noisy and not (_RUNTIME_DEBUG_LOGGING or force):
        return

    logger.info(message)
    if log_file:
        log_file.write(message + "\n")



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
        model.language_model.config.profile_llm_cuda = bool(cfg.profile_llm_cuda)
        model.language_model.config.measure_llm_flops = True
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
    warped_caches        = None     # keyframe KV를 현재 frame으로 warp한 것


    # Rolling-anchor state: previous LLM call cache/image/pose.
    # When cfg.use_rolling_anchor=True, these three are used together as the KV source
    # and geometric warp source, so cache source and warp source stay consistent.
    cache_anchor_cache   = None
    cache_anchor_T_w_c   = None
    cache_anchor_img     = None

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
    
    all_visual_token_indices = list(
        range(fixed_token_start, fixed_token_start + num_patches_per_image)
    ) + list(
        range(wrist_token_start, wrist_token_start + num_patches_per_image)
    )
    episode_total_wall_ms = 0.0
    episode_all_steps = 0

    # Authoritative control frequency:
    episode_control_start = None
    episode_control_end = None
    episode_control_actions = 0

    sim_handle = None
    warp_tracker_initialized = False
    rotary_emb_module = (
        model.language_model.model.layers[0].self_attn.rotary_emb
        if cfg.use_dynam_cache else None
    )

    
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
            if episode_control_start is None:
                episode_control_start = step_start_time

            will_call_llm = len(action_queue) == 0
            need_images = will_call_llm or cfg.save_rollout_videos or cfg.save_runtime_debug_images
            if need_images:
                observation, fixed_img = prepare_observation(obs, resize_size)
                wrist_img = observation["wrist_image"]
            else:
                observation = None
                fixed_img = None
                wrist_img = None
            _t_step1 = time.perf_counter()

            is_bootstrap_call = False
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

            fixed_task_relevant_indices = []
            wrist_task_relevant_indices = []
            fixed_static_indices = None
            wrist_static_indices = None
            static_indices = None
            hole_ratio = None

            kv_warp_mapping = None

            wrist_task_relevant_patch_indices = []
            fixed_task_relevant_patch_indices = []
            wrist_attention_critical_patch_indices = set()

            reusable_count = 0
            fixed_reusable_count = 0
            wrist_reusable_count = 0

            fixed_recency_blocked_count = 0
            wrist_recency_blocked_count = 0
            fixed_recency_blocked_ratio = 0.0
            wrist_recency_blocked_ratio = 0.0

            if cfg.use_dynam_cache and will_call_llm:

                next_llm_call_count = llm_call_count + 1
                is_bootstrap_call = cache_anchor_cache is None

                if sim_handle is None:
                    sim_handle = get_sim_handle(env)

                if not warp_tracker_initialized:
                    warp_tracker.init_env_info(sim_handle, wrist_img.shape)
                    warp_tracker_initialized = True
                T_w_c_curr = get_cam_T_w_c(sim_handle, warp_tracker.cam_id)

                if cache_anchor_T_w_c is None:
                    cache_anchor_T_w_c, cache_anchor_img = T_w_c_curr, wrist_img
    
                # -----------------------------------------------------------------
                # [Step 2] Warping
                # -----------------------------------------------------------------
                final_reusable_indices   = None
                task_relevant_indices    = None
                fixed_task_relevant_indices = []
                wrist_task_relevant_indices = []
                static_indices           = None
                warped_ema_attention_map = None
                
                if is_bootstrap_call:
                    # First LLM call only: bootstrap cache with a full recompute
                    warped_caches = None
                    final_reusable_indices = None
                    applied_reusable_indices = None
                    model.language_model.config.reusable_patches = None

                    if cfg.runtime_debug_logging:
                        log_message(
                            f"[BOOTSTRAP FULL RECOMPUTE] step={t:03d} "
                            f"llm_call={next_llm_call_count} "
                            f"use_recency_cap={cfg.use_recency_cap} "
                            f"T_w_c_curr_none={T_w_c_curr is None}",
                            log_file,
                        )
                else:
                    # Rolling-anchor KV warp: previous LLM call -> current LLM call.
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
                        and kv_warp_mapping is not None
                        and warp_tracker.last_within_bounds is not None
                    ):

                        warped_ema_attention_map = warp_attention_map_with_mapping(
                            prev_attn_map=last_caches[prev_wrist_attention_key],
                            warp_mapping=kv_warp_mapping,
                            within_bounds=warp_tracker.last_within_bounds,
                        )

                        if warped_ema_attention_map is not None:
                            last_caches["latest_warped_wrist_attn_map"] = warped_ema_attention_map.detach().cpu()
                    else:
                        warped_ema_attention_map = None

                # -----------------------------------------------------------------
                # [Step 3] Reuse patch 선정
                # -----------------------------------------------------------------
                
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
                source_cache = cache_anchor_cache

                if cfg.disable_kv_cache_reuse or source_cache is None:
                    warped_caches = None
                elif cfg.reuse_mode in ("fixed_only", "none_prune_off"):
                    warped_caches = clone_cache(source_cache)
                else:
                    if kv_warp_mapping is None:
                        warped_caches = None
                    else:
                        warped_caches = apply_kv_cache_warp(
                            past_key_values_input=source_cache,
                            warp_mapping=kv_warp_mapping,
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

                        fixed_recency_blocked_count, wrist_recency_blocked_count = count_tokens_by_view(
                            list(recency_blocked_candidates),
                            fixed_token_start,
                            wrist_token_start,
                            num_patches_per_image,
                        )

                        fixed_recency_blocked_ratio = fixed_recency_blocked_count / num_patches_per_image * 100
                        wrist_recency_blocked_ratio = wrist_recency_blocked_count / num_patches_per_image * 100

                        if recency_blocked_candidates:
                            candidates = [idx for idx in candidates if idx not in recency_blocked_all]

                            if cfg.runtime_debug_logging:
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
                                adaptive_cam_rot_delta_rad = rotation_delta_angle_rad(prev_signal_T_w_c, T_w_c_curr)
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

                            fixed_age_pre = cache_age_stats(
                                patch_stale_count,
                                fixed_token_start,
                                num_patches_per_image,
                                stale_threshold=cfg.stale_force_threshold,
                            )

                            wrist_age_pre = cache_age_stats(
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
                                if cfg.runtime_debug_logging:
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
    
                applied_reusable_indices = None if is_bootstrap_call else final_reusable_indices    
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
                    model.language_model.config.reusable_patches = indices_tensor 

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

    
                reusable_count = 0 if applied_reusable_indices is None else len(applied_reusable_indices)
                fixed_reusable_count, wrist_reusable_count = count_tokens_by_view(
                    applied_reusable_indices,
                    fixed_token_start,
                    wrist_token_start,
                    num_patches_per_image,
                )
                

            else:
                model.language_model.config.reusable_patches = None
                warped_caches = None
 
            # -----------------------------------------------------------------
            # [Step 4] Observation 준비 + 모델 추론
            #
            # oft 고유 방식: action_queue가 비었을 때만 모델을 호출하고
            # warped_cache / last_caches를 함께 넘겨 KV cache를 재사용.
            # -----------------------------------------------------------------

            if cfg.use_dynam_cache and will_call_llm and (cfg.save_rollout_videos or cfg.save_runtime_debug_images):
                if is_bootstrap_call:
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
                    fixed_reusable_patch_indices = token_indices_to_patch_indices(
                        applied_reusable_indices,
                        fixed_token_start,
                        num_patches_per_image,
                    )
                    wrist_reusable_patch_indices = token_indices_to_patch_indices(
                        applied_reusable_indices,
                        wrist_token_start,
                        num_patches_per_image,
                    )
                    fixed_critical_patch_indices = token_indices_to_patch_indices(
                        fixed_task_relevant_indices,
                        fixed_token_start,
                        num_patches_per_image,
                    )
                    wrist_critical_patch_indices = token_indices_to_patch_indices(
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

            if cfg.save_rollout_videos:
                # will_call_llm 여부와 무관하게 항상 append
                replay_images_heatmap.append(last_heatmap_fixed)
                replay_images_wrist_heatmap.append(last_heatmap_wrist)

            # ── ADAPTIVE PHASE DEBUG IMAGE ─────────────────────────
            if (
                cfg.use_dynam_cache
                and will_call_llm
                and cfg.use_adaptive_pruning
                and cfg.save_runtime_debug_images
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

                if cfg.runtime_debug_logging:
                    log_message(
                        f"[ADAPTIVE PHASE DEBUG IMAGE] step={t:03d} "
                        f"llm_call={next_llm_call_count} "
                        f"path={adaptive_debug_path}",
                        log_file,
                    )
            # ───────────────────────────────────────────────────────


            # If action queue is empty, requery model
            if will_call_llm:
                _t_model_start = time.perf_counter()

                if cfg.use_dynam_cache:
                    llm_call_count += 1

                    if not is_bootstrap_call:
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

                    if cfg.runtime_debug_logging:
                        log_message(
                            f"[PREFWD REUSE CHECK] step={t:03d} "
                            f"llm_call={llm_call_count} "
                            f"hard_refresh={is_bootstrap_call} "
                            f"use_recency_cap={cfg.use_recency_cap} "
                            f"reusable_patches_none={model.language_model.config.reusable_patches is None} "
                            f"warped_caches_none={warped_caches is None} "
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
                        warped_cache=None if is_bootstrap_call else warped_caches,
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

                action_queue.extend(actions)

                if cfg.use_dynam_cache:
                    prev_llm_fixed_img = fixed_img


                # LLaMA 성능 지표 로깅 (실제 추론이 발생한 step에서만)
                _log_llama_perf_metrics(log_file, model, perf_counters=perf_counters, emit_log=cfg.runtime_debug_logging or cfg.profile_llm_cuda),
                        
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

                if not is_bootstrap_call:
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

                        if cfg.runtime_debug_logging:
                            log_message(
                                f"[EXACT REUSE DBG] step={t:03d} "
                                f"last_layer={getattr(llama_model, 'last_reused_patch_step_layer', None)} "
                                f"n_indices={_exact_reused_abs.numel()}",
                                log_file,
                            )

                        if cfg.save_runtime_debug_images:
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
                fixed_age_stats = cache_age_stats(
                    patch_stale_count,
                    fixed_token_start,
                    num_patches_per_image,
                    stale_threshold=cfg.stale_force_threshold,
                )
                wrist_age_stats = cache_age_stats(
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

                    if cfg.runtime_debug_logging:
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

                    if cfg.save_runtime_debug_images:
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

                        if cfg.runtime_debug_logging:
                            log_message(
                                f"[STALE HEATMAP] step={t:03d} llm_call={llm_call_count} path={stale_heatmap_path}",
                                log_file,
                            )

                # ── exact reuse / recompute stats ─────────────────────────
                exact_fixed_reuse_count, exact_wrist_reuse_count = count_tokens_by_view(
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
                    cam_rot_delta_rad = rotation_delta_angle_rad(prev_signal_T_w_c, T_w_c_curr)
                else:
                    cam_trans_delta = np.nan
                    cam_rot_delta_rad = np.nan
                
                if cfg.runtime_debug_logging:
                    log_message("=" * 20 + " SIGNAL " + "=" * 20, log_file)
                    log_message(
                        f"[CALL SIGNAL] step={t:03d} llm_call={llm_call_count} "
                        f"hard_refresh={is_bootstrap_call} "
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
                        f"fixed_recency_blocked_count={fixed_recency_blocked_count} "
                        f"wrist_recency_blocked_count={wrist_recency_blocked_count} "
                        f"fixed_recency_blocked_ratio={fixed_recency_blocked_ratio:.2f}% "
                        f"wrist_recency_blocked_ratio={wrist_recency_blocked_ratio:.2f}% "
                        f"fixed_stale_ge_threshold_count={fixed_stale_ge_threshold_count} "
                        f"fixed_eq{cfg.stale_force_threshold}_index_count={fixed_stale_eq_threshold_count} "
                        f"wrist_eq{cfg.stale_force_threshold}_index_count={wrist_stale_eq_threshold_count} "
                        f"total_eq{cfg.stale_force_threshold}_index_count={total_stale_eq_threshold_count} "
                        f"wrist_stale_ge_threshold_count={wrist_stale_ge_threshold_count} "
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
                    cache_anchor_cache = last_caches

                    if llm_call_count <= 5 or (llm_call_count % 10 == 0):
                        if cfg.runtime_debug_logging:
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
                            if cfg.runtime_debug_logging:
                                log_message(
                                    f"[DBG q_len] layer={item['layer']:02d} "
                                    f"q_after={item['q_after']} "
                                    f"reused_so_far={item['reused_so_far']} "
                                    f"reuse_ratio_actual={item['reuse_ratio_actual']:.2f}% "
                                    f"reuse_ratio_of_candidates={item['reuse_ratio_of_candidates']:.2f}%",
                                    log_file,
                                )

                        elif item["type"] == "summary":
                            if cfg.runtime_debug_logging:
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

            else:
                _t_model_start = time.perf_counter()
                _t_model_end = _t_model_start

            _t_post_model = time.perf_counter()

            # -----------------------------------------------------------------
            # [Step 6] 액션 실행 + control frequency 로깅
            # -----------------------------------------------------------------
                
            # Get action from queue & Process action
            action = process_action(action_queue.popleft(), cfg.model_family)


            # Execute action in environment
            _t_env_start = time.perf_counter()
            obs, reward, done, info = env.step(action.tolist())
            _t_env_end = time.perf_counter()

            episode_control_actions += 1
            episode_control_end = _t_env_end

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

                direction_cosine = safe_direction_cosine(delta_pos, prev_delta_pos)

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
            if cfg.runtime_debug_logging:
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

            if cfg.save_signal_plots:

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

            control_step_time = time.perf_counter() - step_start_time
            episode_control_elapsed = (
                episode_control_end - episode_control_start
                if episode_control_start is not None and episode_control_end is not None
                else 0.0
            )
            current_hz = (
                episode_control_actions / episode_control_elapsed
                if episode_control_elapsed > 0 else 0.0
            )

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
                
            if will_call_llm:

                current_fixed_reuse_ratio = (
                    fixed_reusable_count / num_patches_per_image * 100
                    if will_call_llm and not is_bootstrap_call
                    else 0.0
                )
                current_wrist_reuse_ratio = (
                    wrist_reusable_count / num_patches_per_image * 100
                    if will_call_llm and not is_bootstrap_call
                    else 0.0
                )

                if cfg.runtime_debug_logging:
                    log_message(
                        f"[Control] step={t:03d} | "
                        f"episode_e2e_time_s={episode_control_elapsed:.6f} | "
                        f"episode_e2e_hz={current_hz:.4f} Hz | "
                        f"reusable_patches={reusable_count} | "
                        f"queue_remaining={queue_remaining} | "
                        f"fixed_reuse={current_fixed_reuse_ratio:.2f}% | "
                        f"wrist_reuse={current_wrist_reuse_ratio:.2f}%",
                        log_file,
                    )

                if cfg.save_signal_plots:
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

    if episode_control_start is not None and episode_control_end is not None:
        episode_control_time_s = max(0.0, episode_control_end - episode_control_start)
        perf_counters["real_control_total_time_s"] += episode_control_time_s
        perf_counters["real_control_total_actions"] += episode_control_actions
        episode_hz = (
            episode_control_actions / episode_control_time_s
            if episode_control_time_s > 0 else 0.0
        )
        log_message(
            f"[Episode End-to-End Control] actions={episode_control_actions} "
            f"time_s={episode_control_time_s:.6f} hz={episode_hz:.4f}",
            log_file,
            force=True,
        )

    if cfg.save_signal_plots:
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
            total_episodes=total_episodes, 
            perf_counters=perf_counters,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video with patch overlay already composited on top.
        if cfg.save_rollout_videos:
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
                f"[Authoritative Control Frequency] {avg_real_ctrl_hz:.2f} Hz "
                f"({perf_counters['real_control_total_actions']} executed actions / "
                f"{perf_counters['real_control_total_time_s']:.3f} s) | "
                f"Average LLM Reuse Ratio (Fixed): {fixed_ratio:.2f}% | "
                f"Average LLM Reuse Ratio (Wrist): {wrist_ratio:.2f}%",
                log_file,
            )

            if cfg.runtime_debug_logging and perf_counters["total_all_steps"] > 0:
                avg_step_ms = (
                    perf_counters["total_wall_ms"]
                    / perf_counters["total_all_steps"]
                )
                log_message(
                    f"[Diagnostic Mean Step Latency] {avg_step_ms:.2f} ms/action",
                    log_file,
                )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
        if log_file:
            log_file.flush()

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

    llama_model = getattr(
        getattr(model, "language_model", None),
        "model",
        None,
    )

    if llama_model is not None:
        avg_cuda_latency = getattr(
            llama_model,
            "last_cuda_time_avg",
            None,
        )
        avg_tflops = getattr(
            llama_model,
            "last_tflops_avg",
            None,
        )
        llm_forward_count = getattr(
            llama_model,
            "num_forward",
            0,
        )

        if avg_tflops is not None:
            if avg_cuda_latency is not None:
                log_message(
                    f"[Final LLM Performance] "
                    f"Average CUDA latency: {float(avg_cuda_latency):.6f} ms | "
                    f"Average TFLOPs: {float(avg_tflops):.6f} | "
                    f"LLM forwards: {int(llm_forward_count)}",
                    log_file,
                )
            else:
                log_message(
                    f"[Final LLM Performance] "
                    f"Average TFLOPs: {float(avg_tflops):.6f} | "
                    f"LLM forwards: {int(llm_forward_count)}",
                    log_file,
                )

        

    log_message("=" * 60, log_file)
    
    if cfg.save_signal_plots:
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
