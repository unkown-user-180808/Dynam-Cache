from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def _unwrap_cache(cache_input):
    if isinstance(cache_input, dict):
        return cache_input.get("past_key_values")
    return cache_input


def clone_cache(cache_input):
    actual_cache = _unwrap_cache(cache_input)
    if actual_cache is None:
        return None

    new_cache = DynamicCache()
    for layer_idx in range(len(actual_cache)):
        new_cache.update(
            actual_cache.key_cache[layer_idx].clone(),
            actual_cache.value_cache[layer_idx].clone(),
            layer_idx,
        )

    if isinstance(cache_input, dict):
        result = cache_input.copy()
        result["past_key_values"] = new_cache
        return result
    return new_cache


def apply_kv_cache_warp(
    past_key_values_input,
    warp_mapping,
    rotary_emb_module,
    v_token_start: int = 1,
):
    actual_cache = _unwrap_cache(past_key_values_input)
    if actual_cache is None or len(actual_cache) == 0:
        return None

    device = actual_cache.key_cache[0].device
    warp_mapping = torch.as_tensor(
        warp_mapping,
        dtype=torch.long,
        device=device,
    ).reshape(-1)

    num_visual_tokens = int(warp_mapping.numel())
    v_token_end = v_token_start + num_visual_tokens
    curr_position_ids = torch.arange(
        v_token_start,
        v_token_end,
        device=device,
    ).unsqueeze(0)
    past_position_ids = (warp_mapping + v_token_start).unsqueeze(0)

    rope_ref = actual_cache.key_cache[0].to(torch.float32)
    cos_past, sin_past = rotary_emb_module(rope_ref, past_position_ids)
    cos_curr, sin_curr = rotary_emb_module(rope_ref, curr_position_ids)
    cos_past = cos_past.to(torch.float32)
    sin_past = sin_past.to(torch.float32)
    cos_curr = cos_curr.to(torch.float32)
    sin_curr = sin_curr.to(torch.float32)

    new_cache = DynamicCache()
    for layer_idx in range(len(actual_cache)):
        k_tensor = actual_cache.key_cache[layer_idx]
        v_tensor = actual_cache.value_cache[layer_idx]

        k_prefix = k_tensor[:, :, :v_token_start, :]
        v_prefix = v_tensor[:, :, :v_token_start, :]
        k_visual = k_tensor[:, :, v_token_start:v_token_end, :]
        v_visual = v_tensor[:, :, v_token_start:v_token_end, :]
        k_suffix = k_tensor[:, :, v_token_end:, :]
        v_suffix = v_tensor[:, :, v_token_end:, :]

        gather_index = warp_mapping.view(1, 1, -1, 1).expand(
            k_visual.shape[0],
            k_visual.shape[1],
            -1,
            k_visual.shape[-1],
        )
        k_visual_gathered = torch.gather(k_visual, dim=2, index=gather_index)
        v_visual_warped = torch.gather(v_visual, dim=2, index=gather_index)

        k_fp32 = k_visual_gathered.to(torch.float32)
        dummy_q = torch.zeros_like(k_fp32)
        _, k_raw = apply_rotary_pos_emb(
            dummy_q,
            k_fp32,
            cos_past,
            -sin_past,
            unsqueeze_dim=1,
        )
        _, k_visual_warped = apply_rotary_pos_emb(
            dummy_q,
            k_raw,
            cos_curr,
            sin_curr,
            unsqueeze_dim=1,
        )

        k_new = torch.cat(
            [k_prefix, k_visual_warped.to(k_tensor.dtype), k_suffix],
            dim=2,
        )
        v_new = torch.cat([v_prefix, v_visual_warped, v_suffix], dim=2)
        new_cache.update(k_new, v_new, layer_idx)

    if isinstance(past_key_values_input, dict):
        result = past_key_values_input.copy()
        result["past_key_values"] = new_cache
        return result
    return new_cache


def count_tokens_by_view(
    token_indices: Optional[Iterable[int]],
    fixed_token_start: int,
    wrist_token_start: int,
    num_patches_per_image: int,
) -> Tuple[int, int]:
    if not token_indices:
        return 0, 0

    fixed_token_end = fixed_token_start + num_patches_per_image
    wrist_token_end = wrist_token_start + num_patches_per_image
    fixed_count = 0
    wrist_count = 0

    for token_idx in token_indices:
        idx = int(token_idx)
        if fixed_token_start <= idx < fixed_token_end:
            fixed_count += 1
        elif wrist_token_start <= idx < wrist_token_end:
            wrist_count += 1

    return fixed_count, wrist_count


def token_indices_to_patch_indices(
    token_indices: Optional[Iterable[int]],
    token_start: int,
    num_patches: int = 256,
):
    if not token_indices:
        return set()

    token_end = token_start + num_patches
    return {
        int(token_idx) - token_start
        for token_idx in token_indices
        if token_start <= int(token_idx) < token_end
    }


def cache_age_stats(
    patch_stale_count: Dict[int, int],
    token_start: int,
    num_patches: int = 256,
    stale_threshold: int = 8,
):
    ages = np.fromiter(
        (
            float(patch_stale_count.get(token_idx, 0))
            for token_idx in range(token_start, token_start + num_patches)
        ),
        dtype=np.float32,
        count=num_patches,
    )
    eq_cap_count = int(np.count_nonzero(ages == stale_threshold))
    ge_cap_count = int(np.count_nonzero(ages >= stale_threshold))

    return {
        "mean": float(ages.mean()),
        "max_age": float(ages.max()),
        "eq_cap_count": eq_cap_count,
        "ge_cap_count": ge_cap_count,
        "eq_cap_ratio": eq_cap_count / num_patches * 100.0,
        "ge_cap_ratio": ge_cap_count / num_patches * 100.0,
    }
