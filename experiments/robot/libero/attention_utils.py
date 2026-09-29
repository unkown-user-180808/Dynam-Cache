# attention_utils.py
from typing import Tuple, Dict, Iterable

import torch
import numpy as np
import torch.nn.functional as F
from collections import deque

#################################################################
######################## 유틸리티 함수 ############################
#################################################################
##레이어별 재사용률
@torch.no_grad()


def normalize_attention_layer_ids(layer_ids: Tuple[int, ...]) -> Tuple[int, ...]:
    if layer_ids is None or len(layer_ids) == 0:
        raise ValueError("`attention_layer_ids` must contain at least one layer id.")
    return tuple(int(layer_id) for layer_id in layer_ids)


def spatial_scores_to_map(attn_scores, side=16, device=None):
    """
    [256] attention score를 2D map으로 바꾼다.
    """
    if not isinstance(attn_scores, torch.Tensor):
        attn_scores = torch.as_tensor(attn_scores, dtype=torch.float32)
    attn_scores = attn_scores.detach().to(dtype=torch.float32)
    if device is not None:
        attn_scores = attn_scores.to(device)
    return attn_scores.reshape(1, 1, side, side)

# numpy 구성으로 넘겨주는 함수.
def flatten_attention_scores(attn_scores):
    if isinstance(attn_scores, torch.Tensor):
        attn_scores = attn_scores.detach().to(dtype=torch.float32).cpu().numpy()
    return np.asarray(attn_scores, dtype=np.float32).reshape(-1)


#################################################################
################ attention map 제작 함수 #########################
#################################################################

def token_attention_merge(
    multihead_attention,
    layer_ids=(1,),
    kept_query_positions=None,
    key_token_start=1,
    num_key_tokens=256,
    query_token_start=None,
    query_token_end=None,
    query_row_indices=None,   # ← 새로 추가: 명시적 절대 위치 리스트. 있으면 start/end 무시.
):
    key_token_end = key_token_start + num_key_tokens
    if query_token_start is None:
        query_token_start = key_token_end

    layer_scores = []

    for current_layer_id in layer_ids:
        if current_layer_id >= len(multihead_attention) or multihead_attention[current_layer_id] is None:
            layer_scores.append(torch.zeros(num_key_tokens, dtype=torch.float32))
            continue

        attn_map = multihead_attention[current_layer_id].to(torch.float32).squeeze(0).mean(dim=0)
        q_len, k_len = attn_map.shape

        local_key_token_end = min(key_token_end, k_len)
        if local_key_token_end <= key_token_start:
            score = torch.zeros(num_key_tokens, dtype=torch.float32, device=attn_map.device)
            layer_scores.append(score)
            continue

        if query_row_indices is not None:
            # 명시적 row 선택 (예: stopword 제외된 text row들)
            row_idx_t = torch.as_tensor(query_row_indices, dtype=torch.long, device=attn_map.device)
            row_idx_t = row_idx_t[(row_idx_t >= 0) & (row_idx_t < q_len)]
            if row_idx_t.numel() == 0:
                score = torch.zeros(num_key_tokens, dtype=torch.float32, device=attn_map.device)
                layer_scores.append(score)
                continue
            relation = attn_map[row_idx_t, key_token_start:local_key_token_end]
        elif kept_query_positions is None:
            local_query_end = q_len if query_token_end is None else min(int(query_token_end), q_len)
            relation = attn_map[query_token_start:local_query_end, key_token_start:local_key_token_end]
        else:
            kept_query_positions_t = kept_query_positions.to(attn_map.device)
            assert kept_query_positions_t.numel() == q_len, (
                f"Mismatch: q_len={q_len}, kept_query_positions={kept_query_positions_t.numel()}"
            )
            query_row_mask = kept_query_positions_t >= query_token_start
            if query_token_end is not None:
                query_row_mask = query_row_mask & (kept_query_positions_t < query_token_end)
            relation = attn_map[query_row_mask, key_token_start:local_key_token_end]

        if relation.numel() == 0:
            score = torch.zeros(num_key_tokens, dtype=torch.float32, device=attn_map.device)
        else:
            score = torch.nan_to_num(relation.mean(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
            if score.numel() < num_key_tokens:
                score = F.pad(score, (0, num_key_tokens - score.numel()))
            elif score.numel() > num_key_tokens:
                score = score[:num_key_tokens]
            score = score / score.sum().clamp_min(1e-8)

        layer_scores.append(score)

    if not layer_scores:
        spatial_scores = torch.zeros(num_key_tokens, dtype=torch.float32)
    else:
        spatial_scores = torch.stack(layer_scores, dim=0).mean(dim=0)

    return spatial_scores.cpu()

def token_attention_merge_word_groups(
    multihead_attention,
    word_groups,
    layer_ids=(1,),
    kept_query_positions=None,
    key_token_start=1,
    num_key_tokens=256,
):
    """
    word group 단위로 text→vision attention map을 만든다.

    각 word group 내부 subword rows는 먼저 평균내고,
    그다음 word group끼리 평균낸다.

    이렇게 하면:
        drawer = ['▁dra', 'wer']
    가 token 2개라고 해서 open보다 2배 weight를 갖지 않는다.
    """
    group_scores = []

    for group in word_groups:
        rows = group["rows"]

        score = token_attention_merge(
            multihead_attention=multihead_attention,
            layer_ids=layer_ids,
            kept_query_positions=kept_query_positions,
            key_token_start=key_token_start,
            num_key_tokens=num_key_tokens,
            query_row_indices=rows,
        )

        score = torch.as_tensor(score, dtype=torch.float32)

        if float(score.sum()) > 1e-8:
            group_scores.append(score)

    if len(group_scores) == 0:
        return torch.zeros(num_key_tokens, dtype=torch.float32)

    merged = torch.stack(group_scores, dim=0).mean(dim=0)
    merged = merged / merged.sum().clamp_min(1e-8)

    return merged.cpu()


def update_attention_ema(prev_map, current_map, alpha: float):
    if current_map is None:
        return prev_map
    if prev_map is None:
        return current_map.detach().cpu()

    prev = prev_map.to(device=current_map.device, dtype=torch.float32)
    curr = current_map.to(dtype=torch.float32)

    if prev.shape != curr.shape:
        return current_map.detach().cpu()

    ema = alpha * curr + (1.0 - alpha) * prev
    return ema.detach().cpu()

# 일반적인 영어 기능어 (관사/전치사/접속사/조동사 위주)
STOPWORDS = {
    "the", "a", "an", "of", "at", "and", "or", 
    "is", "are", "be", "this", "that", "for", "it", "its",
}

def get_content_word_row_groups(text_token_strs, text_token_start, stopwords=None):
    """
    SentencePiece token들을 word 단위로 묶은 뒤,
    task instruction span 안의 content word group을 반환한다.

    return 예시:
    [
        {"word": "open", "tokens": ["▁open"], "rows": [523]},
        {"word": "middle", "tokens": ["▁middle"], "rows": [525]},
        {"word": "drawer", "tokens": ["▁dra", "wer"], "rows": [526, 527]},
        {"word": "cabinet", "tokens": ["▁cabinet"], "rows": [531]},
    ]
    """
    if stopwords is None:
        stopwords = STOPWORDS

    def clean_token(tok: str):
        return str(tok).lstrip("▁").strip().lower()

    cleaned_tokens = [clean_token(tok) for tok in text_token_strs]

    # ------------------------------------------------------------
    # 1) task span 찾기: "take to" 뒤부터
    # ------------------------------------------------------------
    task_start = 0
    for i in range(len(cleaned_tokens) - 1):
        if cleaned_tokens[i] == "take" and cleaned_tokens[i + 1] == "to":
            task_start = i + 2
            break

    # ------------------------------------------------------------
    # 2) task span 끝 찾기: ?, newline, Out 전까지
    # ------------------------------------------------------------
    task_end = len(text_token_strs)
    for i in range(task_start, len(cleaned_tokens)):
        clean = cleaned_tokens[i]
        raw = str(text_token_strs[i]).lower()

        if clean in {"?", ".", "!", "out", "out:"}:
            task_end = i
            break

        if "0x0a" in raw or raw == "\n":
            task_end = i
            break

    # ------------------------------------------------------------
    # 3) task span 안에서 SentencePiece token들을 word group으로 묶기
    # ------------------------------------------------------------
    word_groups = []
    current_tokens = []
    current_rows = []

    def flush_current():
        if not current_tokens:
            return

        word = "".join(clean_token(tok) for tok in current_tokens)

        # punctuation/special 제외
        if not word.isalpha():
            return

        # stopword 제외
        if word in stopwords:
            return

        word_groups.append({
            "word": word,
            "tokens": list(current_tokens),
            "rows": list(current_rows),
        })

    for i in range(task_start, task_end):
        tok = str(text_token_strs[i])
        clean = clean_token(tok)

        # punctuation/special token이면 현재 word 닫고 skip
        if not clean.isalpha():
            flush_current()
            current_tokens = []
            current_rows = []
            continue

        is_word_start = tok.startswith("▁")

        if is_word_start:
            flush_current()
            current_tokens = [tok]
            current_rows = [text_token_start + i]
        else:
            # subword continuation
            if current_tokens:
                current_tokens.append(tok)
                current_rows.append(text_token_start + i)
            else:
                # 혹시 문장 처음이 subword로 시작하는 이상 케이스
                current_tokens = [tok]
                current_rows = [text_token_start + i]

    flush_current()

    return word_groups

#################################################################
################ reusable patch sorting methods #################
#################################################################

def sort_candidates_by_attn(candidates, attn_map, token_start, num_patches):
    """candidate indices를 attention 낮은 순(덜 중요한 순)으로 정렬."""
    if attn_map is None or len(candidates) == 0:
        return list(candidates)
    flat = attn_map.flatten()
    scored = []
    for idx in candidates:
        patch_idx = idx - token_start
        score = float(flat[patch_idx]) if 0 <= patch_idx < len(flat) else 0.0
        scored.append((idx, score))
    return [idx for idx, _ in sorted(scored, key=lambda x: x[1])]


def rank_normalize_attn(candidates, attn_map, token_start, num_patches):
    """
    view 내에서 attention score를 rank 기반으로 0~1 정규화.
    반환: {idx: normalized_rank} dict (0=가장 덜 중요, 1=가장 중요)
    """
    if attn_map is None or len(candidates) == 0:
        return {idx: 0.5 for idx in candidates}
    flat = attn_map.flatten()
    scores = []
    for idx in candidates:
        patch_idx = idx - token_start
        score = float(flat[patch_idx]) if 0 <= patch_idx < len(flat) else 0.0
        scores.append((idx, score))
    # attention 낮은 순으로 rank 부여
    sorted_by_score = sorted(scores, key=lambda x: x[1])
    n = max(len(sorted_by_score) - 1, 1)
    return {idx: i / n for i, (idx, _) in enumerate(sorted_by_score)}


def order_candidates_for_pruning(
    fixed_candidates,
    wrist_candidates,
    fixed_attn_map,
    wrist_attn_map,
    fixed_token_start,
    wrist_token_start,
    num_patches_per_image,
    mode="normattn_global",
    progressive_drop_ratios=None,
):
    """
    Sort both-view reuse candidates using the final release policy:
    normalize attention ranks within each view, then globally sort from
    least important to most important.

    The release evaluator no longer supports legacy candidate-ordering
    experiment modes. Any provided mode is ignored and the final
    normattn_global behavior is always used.
    """
    _ = mode
    _ = progressive_drop_ratios
    _ = num_patches_per_image

    fixed_ranks = rank_normalize_attn(
        fixed_candidates,
        fixed_attn_map,
        fixed_token_start,
        num_patches_per_image,
    )
    wrist_ranks = rank_normalize_attn(
        wrist_candidates,
        wrist_attn_map,
        wrist_token_start,
        num_patches_per_image,
    )

    all_candidates = list(fixed_candidates) + list(wrist_candidates)
    all_ranks = {**fixed_ranks, **wrist_ranks}

    return sorted(
        all_candidates,
        key=lambda idx: all_ranks.get(idx, 0.5),
    )


#################################################################
#################################################################

def total_ratio_targets_to_candidate_ratios(
    target_total_ratios,
    n_candidates: int,
    ref_total_tokens: int = 600,
):
    """
    전체 token 기준 target pruning ratio를
    현재 reusable candidate list 기준 ratio로 변환한다.

    target_total_ratios:
        예: (0.08, 0.20, 0.31)
        전체 token 600 기준이면 target count는 (48, 120, 186)

    n_candidates:
        현재 normattn_global ordering으로 만들어진 reusable candidate 개수.

    return:
        dynamic_ratios:
            modeling_llama.py에 넣을 progressive_drop_ratios.
            즉 candidate list 기준 ratio.
        target_counts:
            실제 목표 pruning count tuple.
    """
    if n_candidates is None or int(n_candidates) <= 0:
        return None, (0, 0, 0)

    n_candidates = int(n_candidates)

    t2, t6, t10 = [
        int(round(float(r) * float(ref_total_tokens)))
        for r in target_total_ratios
    ]

    # 현재 candidate 수보다 많이 prune할 수 없으므로 clamp
    t2 = min(t2, n_candidates)
    t6 = min(t6, n_candidates)
    t10 = min(t10, n_candidates)

    # progressive pruning이므로 monotonic 보장
    t6 = max(t6, t2)
    t10 = max(t10, t6)

    dynamic_ratios = [
        float(t2) / float(n_candidates),
        float(t6) / float(n_candidates),
        float(t10) / float(n_candidates),
    ]

    dynamic_ratios = [
        min(1.0, max(0.0, float(r)))
        for r in dynamic_ratios
    ]

    return dynamic_ratios, (t2, t6, t10)



#################################################################
################ attention map 워핑 함수 #########################
#################################################################

def warp_attention_map_with_mapping(
    prev_attn_map,
    warp_mapping=None,
    within_bounds=None,
    fill_value=0.0,
    device=None,
    src_coords=None,
    feat_shape=(16, 16),
    mode="nearest",
):
    """
    prev_attn_map:
        previous attention map.
        shape can be [1, 1, 16, 16], [16, 16], or flat [256].
    warp_mapping:
        shape [256], mapping current_idx -> past_idx
    within_bounds:
        optional boolean mask of shape [256]
    fill_value:
        value for invalid/OOB current patches
    return:
        warped attention map of shape [1, 1, 16, 16]
    """
    if isinstance(prev_attn_map, torch.Tensor):
        prev_map = prev_attn_map.detach().to(dtype=torch.float32)
        src_device = prev_map.device
    else:
        prev_map = torch.as_tensor(prev_attn_map, dtype=torch.float32)
        src_device = torch.device("cpu")

    out_device = device if device is not None else src_device
    prev_map = prev_map.to(out_device)

    if prev_map.dim() == 1:
        prev_map = prev_map.view(1, 1, *feat_shape)
    elif prev_map.dim() == 2:
        prev_map = prev_map.view(1, 1, *feat_shape)

    # continuous warp path
    if src_coords is not None:
        if isinstance(src_coords, torch.Tensor):
            src_coords_t = src_coords.detach().to(dtype=torch.float32, device=out_device)
        else:
            src_coords_t = torch.as_tensor(src_coords, dtype=torch.float32, device=out_device)

        feat_h, feat_w = int(feat_shape[0]), int(feat_shape[1])
        src_x = src_coords_t[0]
        src_y = src_coords_t[1]

        grid_x = (src_x / (feat_w - 1)) * 2 - 1
        grid_y = (src_y / (feat_h - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, feat_h, feat_w, 2)

        warped_map = F.grid_sample(
            prev_map,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True,
        )
        return warped_map

    # discrete mapping path
    prev_scores = prev_map.reshape(-1)

    if isinstance(warp_mapping, torch.Tensor):
        warp_mapping_t = warp_mapping.detach().long().reshape(-1).to(prev_scores.device)
    else:
        warp_mapping_t = torch.as_tensor(
            warp_mapping, dtype=torch.long, device=prev_scores.device
        ).reshape(-1)

    warped_scores = prev_scores[warp_mapping_t].clone()

    if within_bounds is not None:
        if isinstance(within_bounds, torch.Tensor):
            within_bounds_t = within_bounds.detach().to(
                dtype=torch.bool, device=warped_scores.device
            ).reshape(-1)
        else:
            within_bounds_t = torch.as_tensor(
                within_bounds, dtype=torch.bool, device=warped_scores.device
            ).reshape(-1)
        warped_scores[~within_bounds_t] = float(fill_value)

    warped_map = warped_scores.to(device=out_device, dtype=torch.float32).view(1, 1, *feat_shape)
    return warped_map


#################################################################
################ critical patch 뽑는 함수 #########################
#################################################################

def compute_topk_patch_indices(attn_scores, top_k=100):
    """Protect the top-k highest-attention visual patches (A1/VLA-Cache-style)."""
    flat_scores = flatten_attention_scores(attn_scores)
    if flat_scores.size == 0 or int(top_k) <= 0:
        return []
    k = min(int(top_k), int(flat_scores.size))
    order = np.argsort(-flat_scores, kind="stable")[:k]
    return [int(idx) for idx in order]


def compute_critical_patch_indices(attn_scores, zscore_k=0.20):
    """attention score 평균 + zscore_k*std 이상인 패치 인덱스를 반환."""
    flat_scores = flatten_attention_scores(attn_scores)
    mean = float(flat_scores.mean())
    std = float(flat_scores.std())
    if std < 1e-8:
        return [int(np.argmax(flat_scores))]
    threshold = mean + float(zscore_k) * std
    selected = np.flatnonzero(flat_scores >= threshold).tolist()
    if not selected:
        selected = [int(np.argmax(flat_scores))]
    return [int(idx) for idx in selected]



#################################################################
################ reusable patch 개수 제한 걸기 ####################
#################################################################
def apply_view_budget_cap(
    candidates,
    attn_map,
    token_start,
    num_patches_per_image,
    max_ratio,
):
    """
    candidate 중에서 attention 높은(중요한) 것부터 제외하고,
    attention 낮은 순으로 최대 max_ratio * num_patches_per_image 개수만 남긴다.
    max_ratio가 1.0이면 cap 없음(원본 그대로).
    """
    if max_ratio is None or max_ratio >= 1.0:
        return list(candidates)
    if len(candidates) == 0:
        return []

    max_count = int(max_ratio * num_patches_per_image)
    if len(candidates) <= max_count:
        return list(candidates)

    # attention 낮은 순으로 정렬 후 max_count만 남김
    sorted_cands = sort_candidates_by_attn(candidates, attn_map, token_start, num_patches_per_image)
    return sorted_cands[:max_count]

#################### entropy calculation ########################
@torch.no_grad()

@torch.no_grad()
def token_attention_word_group_stats(
    multihead_attention,
    word_groups,
    layer_ids=(1,),
    kept_query_positions=None,
    key_token_start=1,
    num_key_tokens=256,
):
    """Return per-word visual mass before normalization and a spatial map."""
    key_token_end = key_token_start + num_key_tokens
    results = []

    for group in word_groups:
        layer_raw_scores = []

        for layer_id in layer_ids:
            if (
                layer_id >= len(multihead_attention)
                or multihead_attention[layer_id] is None
            ):
                continue

            attn = (
                multihead_attention[layer_id]
                .to(torch.float32)
                .squeeze(0)
                .mean(dim=0)
            )
            q_len, k_len = attn.shape
            local_key_end = min(key_token_end, k_len)
            if local_key_end <= key_token_start:
                continue

            rows = torch.as_tensor(
                group["rows"],
                dtype=torch.long,
                device=attn.device,
            )
            rows = rows[(rows >= 0) & (rows < q_len)]
            if rows.numel() == 0:
                continue

            relation = attn[rows, key_token_start:local_key_end]
            raw_score = torch.nan_to_num(
                relation.mean(dim=0),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if raw_score.numel() < num_key_tokens:
                raw_score = F.pad(
                    raw_score,
                    (0, num_key_tokens - raw_score.numel()),
                )
            elif raw_score.numel() > num_key_tokens:
                raw_score = raw_score[:num_key_tokens]

            layer_raw_scores.append(raw_score)

        if not layer_raw_scores:
            continue

        raw_score = torch.stack(layer_raw_scores, dim=0).mean(dim=0)
        vision_mass = float(raw_score.sum().item())
        spatial_map = (
            raw_score / raw_score.sum().clamp_min(1e-8)
        ).detach().cpu()

        results.append(
            {
                "word": str(group["word"]),
                "tokens": list(group["tokens"]),
                "rows": list(group["rows"]),
                "vision_mass": vision_mass,
                "spatial_map": spatial_map,
            }
        )

    return results


def merge_fixed_wrist_word_stats(fixed_stats, wrist_stats):
    fixed_by_word = {item["word"]: item for item in fixed_stats}
    wrist_by_word = {item["word"]: item for item in wrist_stats}
    merged = []

    for word in sorted(set(fixed_by_word) | set(wrist_by_word)):
        fixed = fixed_by_word.get(word)
        wrist = wrist_by_word.get(word)
        fixed_mass = float(fixed["vision_mass"]) if fixed else 0.0
        wrist_mass = float(wrist["vision_mass"]) if wrist else 0.0
        merged.append(
            {
                "word": word,
                "fixed_vision_mass": fixed_mass,
                "wrist_vision_mass": wrist_mass,
                "vision_mass": fixed_mass + wrist_mass,
                "fixed_spatial_map": fixed["spatial_map"] if fixed else None,
                "wrist_spatial_map": wrist["spatial_map"] if wrist else None,
            }
        )

    return merged