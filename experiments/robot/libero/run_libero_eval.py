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
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark
import time
import wandb

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
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
    count_tokens_by_view,
)
from experiments.robot.libero.warp_utils import (
    WarpTracker,
    compute_static_patch_indices,
    get_cam_T_w_c,
    get_sim_handle,
)
from experiments.robot.libero.attention_utils import (
    warp_attention_map_with_mapping,
    compute_critical_patch_indices,
    order_candidates_for_pruning,
)

from experiments.robot.libero.kinematic_budget_utils import (
    detect_approach_signal,
)

class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

_CONFIG_LOG_KEYS = (
    "model_family",
    "pretrained_checkpoint",
    "task_suite_name",
    "num_trials_per_task",
    "seed",
    "num_open_loop_steps",
    "use_dynam_cache",
    "profile_llm_cuda",
)


 
@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    use_dynam_cache: bool = True

    run_id_note: Optional[str] = None
    memo: Optional[str] = None
    local_log_dir: str = "./experiments/logs"

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 7
    max_total_episodes: int = 0
    fixed_run_id: Optional[str] = None

    profile_llm_cuda: bool = True

 
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

def _accumulate_pruning_metrics(model, perf_counters) -> None:
    language_model = getattr(model, "language_model", None)
    llama_model = getattr(language_model, "model", None)

    if llama_model is None:
        return

    q_ratios = getattr(llama_model, "last_q_ratios", None)
    if not q_ratios:
        return

    for layer_idx, values in q_ratios.items():
        if layer_idx == "total":
            continue

        perf_counters["layer_calls"][layer_idx] = (
            perf_counters["layer_calls"].get(layer_idx, 0) + 1
        )
        perf_counters["layer_pruned_tokens"][layer_idx] = (
            perf_counters["layer_pruned_tokens"].get(layer_idx, 0)
            + values["pruned_tokens"]
        )
        perf_counters["layer_original_tokens"][layer_idx] = (
            perf_counters["layer_original_tokens"].get(layer_idx, 0)
            + values["original_seq_length"]
        )
        perf_counters["layer_fixed_pruned"][layer_idx] = (
            perf_counters["layer_fixed_pruned"].get(layer_idx, 0)
            + values.get("fixed_pruned", 0)
        )
        perf_counters["layer_wrist_pruned"][layer_idx] = (
            perf_counters["layer_wrist_pruned"].get(layer_idx, 0)
            + values.get("wrist_pruned", 0)
        )

    total = q_ratios.get("total")
    if total is not None:
        perf_counters["total_pruning_calls"] += 1
        perf_counters["total_pruned_tokens_sum"] += total["pruned_tokens"]
        perf_counters["total_original_tokens_sum"] += total["original_seq_length"]
        perf_counters["total_fixed_pruned_sum"] += total.get("fixed_pruned", 0)
        perf_counters["total_wrist_pruned_sum"] += total.get("wrist_pruned", 0)

def log_message(message: str, log_file=None) -> None:
    logger.info(message)

    if log_file is not None:
        log_file.write(message + "\n")


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop

    assert not (
        cfg.load_in_8bit
        and cfg.load_in_4bit
    )

    valid_suites = [
        suite.value
        for suite in TaskSuite
    ]

    assert cfg.task_suite_name in valid_suites

def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)


    if cfg.model_family == "openvla":
        lm_config = model.language_model.config

        lm_config.profile_llm_cuda = bool(cfg.profile_llm_cuda)
        lm_config.measure_llm_flops = True

        if cfg.use_dynam_cache:
            lm_config.progressive_pruning_layers = list(
                (2, 6, 10)
            )
            lm_config.progressive_drop_ratios = None
            lm_config.debug_progressive_drop = False
            lm_config.collect_attn_layers = list(
                (1,)
            )
        else:
            lm_config.progressive_pruning_layers = None
            lm_config.progressive_drop_ratios = None
            lm_config.debug_progressive_drop = False
            lm_config.collect_attn_layers = None
            lm_config.reusable_patches = None
            lm_config.warped_past_key_values = None
            lm_config.force_cache_output = False
            lm_config.force_attention_output = False

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,
        )

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(
            cfg,
            model.llm_dim,
        )

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(
            cfg,
            model.llm_dim,
        )

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return (
        model,
        action_head,
        proprio_projector,
        noisy_action_projector,
        processor,
    )

def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    unnorm_key = cfg.task_suite_name

    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    cfg.unnorm_key = unnorm_key

def setup_logging(cfg: GenerateConfig):
    run_id = (
        cfg.fixed_run_id
        or f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    )

    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    if cfg.memo is not None and cfg.memo.strip():
        memo_tag = _sanitize_for_filename(cfg.memo)
        if memo_tag:
            run_id += f"--memo_{memo_tag}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)

    local_log_filepath = os.path.join(
        cfg.local_log_dir,
        run_id + ".txt",
    )

    log_file = open(local_log_filepath, "w")

    logger.info(
        f"Logging to local log file: {local_log_filepath}"
    )

    _write_run_header(
        log_file,
        cfg,
        run_id,
    )

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id

def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    
    log_message("Using default initial states", log_file)
    return initial_states, None

def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

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
    action = normalize_gripper_action(action, binarize=True)

    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def build_progressive_reuse_plan(
    *,
    n_candidates: int,
    ref_total_tokens: int,
    fine_manipulation: bool,
):
    target_total_ratios = (
        (0.02, 0.16, 0.20)
        if fine_manipulation
        else (0.26, 0.46, 0.70)
    )

    n_candidates = max(
        0,
        int(n_candidates),
    )

    ref_total_tokens = max(
        1,
        int(ref_total_tokens),
    )

    progressive_drop_ratios = []
    previous_count = 0

    for target_ratio in target_total_ratios:
        target_count = int(
            round(
                float(target_ratio)
                * ref_total_tokens
            )
        )

        target_count = max(
            previous_count,
            target_count,
        )

        target_count = min(
            n_candidates,
            target_count,
        )

        candidate_ratio = (
            target_count / n_candidates
            if n_candidates > 0
            else 0.0
        )

        progressive_drop_ratios.append(
            float(
                np.clip(
                    candidate_ratio,
                    0.0,
                    1.0,
                )
            )
        )

        previous_count = target_count

    return progressive_drop_ratios



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
    """Run one LIBERO episode with optional Dynam-Cache inference."""

    if perf_counters is None:
        perf_counters = {}

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

    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    sim_handle = (
        get_sim_handle(env)
        if cfg.use_dynam_cache
        else None
    )

    action_queue = deque(
        maxlen=cfg.num_open_loop_steps
    )

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        logger.warning(
            "cfg.num_open_loop_steps (%d) does not match "
            "NUM_ACTIONS_CHUNK (%d).",
            cfg.num_open_loop_steps,
            NUM_ACTIONS_CHUNK,
        )

    t = 0
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    last_caches = {}
    previous_action_chunk = []

    warp_tracker = (
        WarpTracker(
            keyframe_interval=0
        )
        if cfg.use_dynam_cache
        else None
    )

    warp_tracker_initialized = False

    rotary_emb_module = (
        model.language_model.model.layers[0]
        .self_attn.rotary_emb
        if cfg.use_dynam_cache
        else None
    )

    cache_anchor_cache = None
    cache_anchor_T_w_c = None
    cache_anchor_img = None

    prev_llm_fixed_img = None

    prev_fixed_attention_key = (
        "latest_fixed_spatial_map"
    )

    prev_wrist_attention_key = (
        "ema_wrist_spatial_map"
    )

    num_patches_per_image = 256
    fixed_token_start = 1
    wrist_token_start = (
        fixed_token_start + num_patches_per_image
    )

    all_visual_token_indices = list(
        range(
            fixed_token_start,
            fixed_token_start + num_patches_per_image,
        )
    ) + list(
        range(
            wrist_token_start,
            wrist_token_start + num_patches_per_image,
        )
    )

    patch_stale_count = {}

    episode_control_start = None
    episode_control_end = None
    episode_control_actions = 0

    success = False

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(
                    get_libero_dummy_action(
                        cfg.model_family
                    )
                )
                t += 1
                continue

            if episode_control_start is None:
                episode_control_start = (
                    time.perf_counter()
                )

            will_call_llm = len(action_queue) == 0

            observation = None
            fixed_img = None
            wrist_img = None
            T_w_c_curr = None

            is_bootstrap_call = False
            warped_caches = None
            applied_reusable_indices = None

            fixed_task_relevant_indices = []
            wrist_task_relevant_indices = []

            if will_call_llm:
                observation, fixed_img = prepare_observation(
                    obs,
                    resize_size,
                )
                wrist_img = observation["wrist_image"]

            if cfg.use_dynam_cache and will_call_llm:
                is_bootstrap_call = (
                    cache_anchor_cache is None
                )

                if sim_handle is None:
                    sim_handle = get_sim_handle(env)

                if not warp_tracker_initialized:
                    warp_tracker.init_env_info(
                        sim_handle,
                        wrist_img.shape,
                    )
                    warp_tracker_initialized = True

                T_w_c_curr = get_cam_T_w_c(
                    sim_handle,
                    warp_tracker.cam_id,
                )

                if is_bootstrap_call:
                    model.language_model.config.reusable_patches = None
                    model.language_model.config.progressive_drop_ratios = None

                else:
                    kv_warp_mapping = None
                    wrist_static_indices = None

                    if (
                        cache_anchor_T_w_c is not None
                        and cache_anchor_img is not None
                    ):
                        (
                            kv_warp_mapping,
                            wrist_static_indices,
                        ) = warp_tracker.step_warp(
                            cache_anchor_T_w_c,
                            cache_anchor_img,
                            T_w_c_curr,
                            wrist_img,
                            img_shape=wrist_img.shape[:2],
                            threshold=(
                                0.40
                            ),
                            use_cosine_similarity=(
                                True
                            ),
                            include_warp_holes=(
                                False
                            ),
                        )

                    fixed_static_indices = None

                    if prev_llm_fixed_img is not None:
                        fixed_static_indices = compute_static_patch_indices(prev_llm_fixed_img, fixed_img, threshold=0.40, use_cosine_similarity=True)

                    prev_wrist_map = last_caches.get(
                        prev_wrist_attention_key
                    )

                    if (
                        prev_wrist_map is not None
                        
                        and kv_warp_mapping is not None
                        and warp_tracker.last_within_bounds is not None
                    ):
                        wrist_attention_map = (
                            warp_attention_map_with_mapping(
                                prev_attn_map=prev_wrist_map,
                                warp_mapping=kv_warp_mapping,
                                within_bounds=(
                                    warp_tracker.last_within_bounds
                                ),
                            )
                        )
                    else:
                        wrist_attention_map = prev_wrist_map

                    wrist_relevant_patches = []

                    if wrist_attention_map is not None:
                        wrist_relevant_patches = compute_critical_patch_indices(wrist_attention_map, zscore_k=0.12)

                    wrist_task_relevant_indices = [
                        wrist_token_start + idx
                        for idx in wrist_relevant_patches
                    ]

                    fixed_map = last_caches.get(
                        prev_fixed_attention_key
                    )

                    if fixed_map is not None:
                        fixed_relevant_patches = compute_critical_patch_indices(fixed_map, zscore_k=0.20)

                        fixed_task_relevant_indices = [
                            fixed_token_start + idx
                            for idx in fixed_relevant_patches
                        ]

                    kinematic_signal = None

                    if previous_action_chunk:
                        kinematic_signal = detect_approach_signal(previous_action_chunk, preview_steps=8, slowdown_threshold=0.18, smoothing_window=3, lead_steps_threshold=4)

                    task_relevant_indices = set(
                        fixed_task_relevant_indices
                        + wrist_task_relevant_indices
                    )

                    static_indices = []

                    if fixed_static_indices is not None:
                        static_indices.extend(
                            fixed_token_start + idx
                            for idx in fixed_static_indices
                        )

                    if wrist_static_indices is not None:
                        static_indices.extend(
                            wrist_token_start + idx
                            for idx in wrist_static_indices
                        )

                    source_cache = cache_anchor_cache

                    if source_cache is None:
                        warped_caches = None

                    elif kv_warp_mapping is not None:
                        warped_caches = apply_kv_cache_warp(
                            past_key_values_input=source_cache,
                            warp_mapping=kv_warp_mapping,
                            rotary_emb_module=rotary_emb_module,
                            v_token_start=wrist_token_start,
                        )

                    else:
                        warped_caches = None

                    candidates = [
                        idx
                        for idx in static_indices
                        if idx
                        not in task_relevant_indices
                    ]

                    if patch_stale_count:
                        stale_tokens = {idx for (idx, age) in patch_stale_count.items() if age >= 8}
                        candidates = [idx for idx in candidates if idx not in stale_tokens]

                    fixed_candidates = [idx for idx in candidates if fixed_token_start <= idx < fixed_token_start + num_patches_per_image]
                    wrist_candidates = [idx for idx in candidates if wrist_token_start <= idx < wrist_token_start + num_patches_per_image]
                    final_reusable_indices = order_candidates_for_pruning(fixed_candidates=fixed_candidates, wrist_candidates=wrist_candidates, fixed_attn_map=last_caches.get(prev_fixed_attention_key), wrist_attn_map=last_caches.get(prev_wrist_attention_key), fixed_token_start=fixed_token_start, wrist_token_start=wrist_token_start, num_patches_per_image=num_patches_per_image, mode='normattn_global')
                    llama_model = model.language_model.model
                    prev_forward_cache = getattr(llama_model, 'last_forward_cache', None)
                    if prev_forward_cache is not None and hasattr(prev_forward_cache, 'get_seq_length'):
                        ref_total_tokens = int(prev_forward_cache.get_seq_length())
                    else:
                        q_total = (getattr(llama_model, 'last_q_ratios', {}) or {}).get('total', {})
                        ref_total_tokens = int(q_total.get('original_seq_length', 600))
                    progressive_drop_ratios = None
                    if kinematic_signal is not None:
                        progressive_drop_ratios = (
                            build_progressive_reuse_plan(
                                n_candidates=len(
                                    final_reusable_indices
                                ),
                                ref_total_tokens=ref_total_tokens,
                                fine_manipulation=bool(
                                    kinematic_signal.active
                                ),
                            )
                        )
                    applied_reusable_indices = final_reusable_indices
                    if not applied_reusable_indices:
                        model.language_model.config.reusable_patches = None
                        model.language_model.config.progressive_drop_ratios = None
                    else:
                        model.language_model.config.reusable_patches = torch.tensor(applied_reusable_indices, dtype=torch.long, device=DEVICE)
                        model.language_model.config.progressive_pruning_layers = list((2, 6, 10))
                        model.language_model.config.progressive_drop_ratios = [float(x) for x in progressive_drop_ratios] if progressive_drop_ratios is not None else None

            else:
                (
                    model.language_model.config
                    .reusable_patches
                ) = None
                (
                    model.language_model.config
                    .progressive_drop_ratios
                ) = None
                warped_caches = None

            if will_call_llm:
                llama_model = (
                    model.language_model.model
                )

                llama_model.last_reused_patch_indices = None
                llama_model.last_q_ratios = {}

                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=(
                        proprio_projector
                    ),
                    noisy_action_projector=(
                        noisy_action_projector
                    ),
                    use_film=cfg.use_film,
                    last_caches=last_caches,
                    warped_cache=(
                        None
                        if is_bootstrap_call
                        else warped_caches
                    ),
                )

                action_queue.extend(actions)

                if cfg.use_dynam_cache:
                    previous_action_chunk = list(actions)
                    prev_llm_fixed_img = fixed_img

                _accumulate_pruning_metrics(
                    model,
                    perf_counters,
                )

                if (
                    cfg.use_dynam_cache
                    and not is_bootstrap_call
                ):
                    reused = getattr(
                        llama_model,
                        "last_reused_patch_indices",
                        None,
                    )

                    if (
                        reused is not None
                        and reused.numel() > 0
                    ):
                        exact_reused_set = {
                            int(idx)
                            for idx
                            in reused.tolist()
                        }
                    else:
                        exact_reused_set = set()

                    for idx in all_visual_token_indices:
                        if idx in exact_reused_set:
                            patch_stale_count[idx] = (
                                patch_stale_count.get(
                                    idx,
                                    0,
                                )
                                + 1
                            )
                        else:
                            patch_stale_count[idx] = 0

                    for idx in (
                        fixed_task_relevant_indices
                        + wrist_task_relevant_indices
                    ):
                        patch_stale_count[idx] = 0

                if cfg.use_dynam_cache:
                    cache_anchor_T_w_c = T_w_c_curr
                    cache_anchor_img = wrist_img
                    cache_anchor_cache = last_caches

            raw_action = action_queue.popleft()

            action = process_action(
                raw_action,
                cfg.model_family,
            )

            obs, _, done, _ = env.step(
                action.tolist()
            )

            episode_control_actions += 1
            episode_control_end = time.perf_counter()

            if done:
                success = True
                break

            t += 1

    except Exception as exc:
        import traceback

        log_message(
            f"Episode error: {exc}",
            log_file,
        )
        log_message(
            traceback.format_exc(),
            log_file,
        )
        raise

    if (
        episode_control_start is not None
        and episode_control_end is not None
    ):
        episode_control_time_s = max(
            0.0,
            episode_control_end
            - episode_control_start,
        )

        perf_counters[
            "real_control_total_time_s"
        ] += episode_control_time_s

        perf_counters[
            "real_control_total_actions"
        ] += episode_control_actions

    return success

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
    """Evaluate one LIBERO task."""

    if perf_counters is None:
        perf_counters = {}

    task = task_suite.get_task(task_id)

    initial_states, all_initial_states = (
        load_initial_states(
            cfg,
            task_suite,
            task_id,
            log_file,
        )
    )

    env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=cfg.env_img_res,
    )

    task_episodes = 0
    task_successes = 0

    for episode_idx in tqdm.tqdm(
        range(cfg.num_trials_per_task)
    ):
        if (
            cfg.max_total_episodes > 0
            and total_episodes
            >= cfg.max_total_episodes
        ):
            break

        log_message(
            f"Task: {task_description}",
            log_file,
        )

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[
                episode_idx
            ]
        else:
            task_key = task_description.replace(
                " ",
                "_",
            )
            episode_key = f"demo_{episode_idx}"

            if not all_initial_states[
                task_key
            ][episode_key]["success"]:
                continue

            initial_state = np.array(
                all_initial_states[
                    task_key
                ][episode_key]["initial_state"]
            )

        success = run_episode(
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

        task_episodes += 1
        total_episodes += 1

        if success:
            task_successes += 1
            total_successes += 1

        log_message(
            f"Success: {success}",
            log_file,
        )

        log_message(
            f"Episodes: {total_episodes} | "
            f"Successes: {total_successes}",
            log_file,
        )

        if log_file is not None:
            log_file.flush()

    task_success_rate = (
        float(task_successes)
        / float(task_episodes)
        if task_episodes > 0
        else 0.0
    )

    log_message(
        f"Task success rate: "
        f"{task_success_rate:.4f}",
        log_file,
    )

    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}":
                    task_success_rate,
                f"num_episodes/{task_description}":
                    task_episodes,
            }
        )

    return total_episodes, total_successes

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Evaluate OpenVLA-OFT with or without Dynam-Cache on LIBERO."""

    validate_config(cfg)

    set_seed_everywhere(cfg.seed)

    (
        model,
        action_head,
        proprio_projector,
        noisy_action_projector,
        processor,
    ) = initialize_model(cfg)

    resize_size = get_image_resize_size(cfg)

    (
        log_file,
        local_log_filepath,
        run_id,
    ) = setup_logging(cfg)

    log_message(
        f"[DYNAM CONFIG] "
        f"Enabled={cfg.use_dynam_cache} | "
        "WristWarp=True | "
        "RecencyCap=8 | "
        "AttentionLayers=(1,) | "
        "Ordering=normattn_global | "
        f"KinematicRegimes=free_motion/fine_manipulation",
        log_file,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[
        cfg.task_suite_name
    ]()

    num_tasks = task_suite.n_tasks

    perf_counters = {
        "real_control_total_time_s": 0.0,
        "real_control_total_actions": 0,
        "layer_pruned_tokens": {},
        "layer_original_tokens": {},
        "layer_fixed_pruned": {},
        "layer_wrist_pruned": {},
        "layer_calls": {},
        "total_pruning_calls": 0,
        "total_pruned_tokens_sum": 0,
        "total_original_tokens_sum": 0,
        "total_fixed_pruned_sum": 0,
        "total_wrist_pruned_sum": 0,
    }

    total_episodes = 0
    total_successes = 0

    for task_id in tqdm.tqdm(
        range(num_tasks)
    ):
        if (
            cfg.max_total_episodes > 0
            and total_episodes
            >= cfg.max_total_episodes
        ):
            break

        (
            total_episodes,
            total_successes,
        ) = run_task(
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

    final_success_rate = (
        float(total_successes)
        / float(total_episodes)
        if total_episodes > 0
        else 0.0
    )

    control_time_s = float(
        perf_counters[
            "real_control_total_time_s"
        ]
    )

    control_actions = int(
        perf_counters[
            "real_control_total_actions"
        ]
    )

    control_hz = (
        control_actions / control_time_s
        if control_time_s > 0
        else 0.0
    )

    log_message(
        "Final results:",
        log_file,
    )
    log_message(
        f"Total episodes: {total_episodes}",
        log_file,
    )
    log_message(
        f"Total successes: {total_successes}",
        log_file,
    )
    log_message(
        f"Overall success rate: "
        f"{final_success_rate:.4f} "
        f"({final_success_rate * 100:.1f}%)",
        log_file,
    )
    log_message(
        f"Average control frequency: "
        f"{control_hz:.4f} Hz",
        log_file,
    )

    if perf_counters["layer_calls"]:
        log_message(
            "[Final Pruning Stats]",
            log_file,
        )

        for layer_idx in sorted(
            perf_counters["layer_calls"]
        ):
            calls = perf_counters[
                "layer_calls"
            ][layer_idx]

            avg_pruned = (
                perf_counters[
                    "layer_pruned_tokens"
                ][layer_idx]
                / calls
            )

            avg_original = (
                perf_counters[
                    "layer_original_tokens"
                ][layer_idx]
                / calls
            )

            avg_fixed = (
                perf_counters[
                    "layer_fixed_pruned"
                ][layer_idx]
                / calls
            )

            avg_wrist = (
                perf_counters[
                    "layer_wrist_pruned"
                ][layer_idx]
                / calls
            )

            log_message(
                f"  L{layer_idx}: "
                f"avg_pruned="
                f"{avg_pruned:.1f}/"
                f"{avg_original:.0f} "
                f"({avg_pruned / avg_original * 100:.1f}%) | "
                f"fixed={avg_fixed:.1f}/256 "
                f"({avg_fixed / 256 * 100:.1f}%) | "
                f"wrist={avg_wrist:.1f}/256 "
                f"({avg_wrist / 256 * 100:.1f}%)",
                log_file,
            )

        calls = perf_counters[
            "total_pruning_calls"
        ]

        if calls > 0:
            avg_pruned = (
                perf_counters[
                    "total_pruned_tokens_sum"
                ]
                / calls
            )

            avg_original = (
                perf_counters[
                    "total_original_tokens_sum"
                ]
                / calls
            )

            avg_fixed = (
                perf_counters[
                    "total_fixed_pruned_sum"
                ]
                / calls
            )

            avg_wrist = (
                perf_counters[
                    "total_wrist_pruned_sum"
                ]
                / calls
            )

            log_message(
                f"  Total: "
                f"avg_pruned="
                f"{avg_pruned:.1f}/"
                f"{avg_original:.0f} "
                f"({avg_pruned / avg_original * 100:.1f}%) | "
                f"fixed={avg_fixed:.1f}/256 "
                f"({avg_fixed / 256 * 100:.1f}%) | "
                f"wrist={avg_wrist:.1f}/256 "
                f"({avg_wrist / 256 * 100:.1f}%)",
                log_file,
            )

    llama_model = getattr(
        getattr(
            model,
            "language_model",
            None,
        ),
        "model",
        None,
    )

    avg_cuda_latency = None
    avg_tflops = None
    llm_forward_count = 0

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
                "[Final LLM Performance] "
                f"Average CUDA latency: "
                f"{float(avg_cuda_latency):.6f} ms | "
                f"Average TFLOPs: "
                f"{float(avg_tflops):.6f} | "
                f"LLM forwards: "
                f"{int(llm_forward_count)}",
                log_file,
            )
        else:
            log_message(
                "[Final LLM Performance] "
                f"Average TFLOPs: "
                f"{float(avg_tflops):.6f} | "
                f"LLM forwards: "
                f"{int(llm_forward_count)}",
                log_file,
            )

    tflops_part = (
        f"TFLOPs={float(avg_tflops):.6f}"
        if avg_tflops is not None
        else "TFLOPs=NA"
    )

    cuda_part = (
        f"CUDA={float(avg_cuda_latency):.2f} ms"
        if avg_cuda_latency is not None
        else "CUDA=NA"
    )

    log_message(
        "[FINAL SUMMARY] "
        f"Suite={cfg.task_suite_name} | "
        f"Episodes={total_episodes} | "
        f"Successes={total_successes} | "
        f"SR={100.0 * final_success_rate:.2f}% | "
        f"{tflops_part} | "
        f"{cuda_part} | "
        f"Control={control_hz:.3f} Hz",
        log_file,
    )

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total":
                    final_success_rate,
                "num_episodes/total":
                    total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    if log_file is not None:
        log_file.close()

    return final_success_rate

if __name__ == "__main__":
    eval_libero()
