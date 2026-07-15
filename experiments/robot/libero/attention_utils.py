# attention_utils.py
from typing import Tuple

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
    mode="original",
    progressive_drop_ratios=None,
):
    """
    both-view reuse candidate를 지정된 mode로 정렬하여 반환.

    mode:
        interleave       - attention 낮은 순으로 각 view 정렬 후 번갈아 배치
        normattn_global  - view-wise rank normalize 후 global sort (낮은 순)
        viewattn_ratioaware - ratio 미리 알고 view별 비율 맞춰 구간 배치
    """

    # 각 view별 attention 낮은 순 정렬
    fixed_sorted = sort_candidates_by_attn(
        fixed_candidates, fixed_attn_map, fixed_token_start, num_patches_per_image
    )
    wrist_sorted = sort_candidates_by_attn(
        wrist_candidates, wrist_attn_map, wrist_token_start, num_patches_per_image
    )

    if mode == "interleave":
        interleaved = []
        for f, w in zip(fixed_sorted, wrist_sorted):
            interleaved.extend([f, w])
        len_min = min(len(fixed_sorted), len(wrist_sorted))
        interleaved += fixed_sorted[len_min:] + wrist_sorted[len_min:]
        return interleaved

    elif mode == "normattn_global":
        fixed_ranks = rank_normalize_attn(
            fixed_candidates, fixed_attn_map, fixed_token_start, num_patches_per_image
        )
        wrist_ranks = rank_normalize_attn(
            wrist_candidates, wrist_attn_map, wrist_token_start, num_patches_per_image
        )
        all_candidates = fixed_candidates + wrist_candidates
        all_ranks = {**fixed_ranks, **wrist_ranks}
        return sorted(all_candidates, key=lambda idx: all_ranks.get(idx, 0.5))

    
    elif mode == "viewattn_ratioaware":
        if not progressive_drop_ratios:
            # ratio 없으면 interleave fallback
            interleaved = []
            for f, w in zip(fixed_sorted, wrist_sorted):
                interleaved.extend([f, w])
            len_min = min(len(fixed_sorted), len(wrist_sorted))
            interleaved += fixed_sorted[len_min:] + wrist_sorted[len_min:]
            return interleaved

        n_fixed = len(fixed_sorted)
        n_wrist = len(wrist_sorted)
        total_cands = n_fixed + n_wrist

        result = []
        fixed_used = 0
        wrist_used = 0
        prev_ratio = 0.0

        for ratio in progressive_drop_ratios:
            n_total_drop = int(ratio * total_cands)
            n_new_drop = n_total_drop - int(prev_ratio * total_cands)
            if n_new_drop <= 0:
                prev_ratio = ratio
                continue
            # fixed/wrist 수 비율에 맞게 이 구간 배분
            n_fixed_drop = int(n_new_drop * n_fixed / max(total_cands, 1))
            n_wrist_drop = n_new_drop - n_fixed_drop
            result += fixed_sorted[fixed_used:fixed_used + n_fixed_drop]
            result += wrist_sorted[wrist_used:wrist_used + n_wrist_drop]
            fixed_used += n_fixed_drop
            wrist_used += n_wrist_drop
            prev_ratio = ratio

        # 남은 것 (끝까지 살아남을 것들)
        result += fixed_sorted[fixed_used:]
        result += wrist_sorted[wrist_used:]
        return result

    # fallback
    return fixed_candidates + wrist_candidates

#################################################################
################ adaptive pruning budget helpers ################
#################################################################

ONLINE_ADAPTIVE_SIGNAL_NAMES = (
    "hole_ratio",
    "cam_trans_delta",
    "cam_rot_delta_rad",

    "recent_eef_speed",
    "recent_speed_drop_ratio",
    "recent_abs_accel",
    "recent_abs_jerk",

    "recent_dgripper_asym",
    "recent_action_eef_ratio",

    "direction_dip_ratio",
    "direction_cosine_std",
)

MOTION_SIGNAL_NAMES = (
    "hole_ratio",
    "cam_trans_delta",
    "cam_rot_delta_rad",
)

TRANSITION_SIGNAL_NAMES = (
    "recent_speed_drop_ratio",
    "recent_abs_accel",
    "recent_abs_jerk",
    "direction_dip_ratio",
    "direction_cosine_std",
)

CONTACT_SIGNAL_NAMES = (
    "recent_dgripper_asym",
    "recent_action_eef_ratio",
)

def interpolate_total_prune_ratios_from_risk(
    risk_score: float,
    min_total_prune_ratios,
    max_total_prune_ratios,
):
    """
    risk_score=0.0 -> max pruning
    risk_score=1.0 -> min pruning
    """
    r = float(np.clip(risk_score, 0.0, 1.0))

    min_ratios = np.asarray(min_total_prune_ratios, dtype=np.float32)
    max_ratios = np.asarray(max_total_prune_ratios, dtype=np.float32)

    target = max_ratios - r * (max_ratios - min_ratios)

    # monotonic 보장
    for i in range(1, len(target)):
        target[i] = max(target[i], target[i - 1])

    return tuple(float(x) for x in target)

def _valid_float_or_none(x):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return x


def _clamp01(x):
    x = _valid_float_or_none(x)
    if x is None:
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _percentile_rank(value, history):
    """
    현재 value가 history 안에서 어느 정도 큰지 0~1로 반환.
    high value = high rank.
    단, high rank가 항상 high risk라는 뜻은 아님.
    eef_speed는 high rank를 transit signal로 해석할 수 있음.
    """
    v = _valid_float_or_none(value)
    if v is None:
        return 0.0

    hist = [_valid_float_or_none(x) for x in history]
    hist = [x for x in hist if x is not None]

    if len(hist) == 0:
        return 0.0

    hist = np.asarray(hist, dtype=np.float32)

    if float(hist.max() - hist.min()) < 1e-8:
        return 0.0

    return float((hist <= v).mean())


def _aggregate_ranks(ranks, mode="top2_mean"):
    vals = [float(v) for v in ranks if v is not None and np.isfinite(float(v))]

    if len(vals) == 0:
        return 0.0

    vals = np.asarray(vals, dtype=np.float32)

    if mode == "max":
        return float(vals.max())

    if mode == "mean":
        return float(vals.mean())

    # default: top2_mean
    vals_sorted = np.sort(vals)[::-1]
    k = min(2, len(vals_sorted))
    return float(vals_sorted[:k].mean())


class OnlineAdaptiveRiskController:
    """
    Warmup-guarded exposure-aware pruning controller.

    핵심:
    - eef_speed high를 risk로 보지 않는다.
    - motion signal은 보조적인 modifier로만 사용한다.
    - exposure는 accelerator가 아니라 brake로 사용한다.
    - 초반 cold-start에서는 aggressive pruning을 막는다.
    - low-speed + gripper/contact/exposure 상황을 precision risk로 본다.
    """

    def __init__(
        self,
        window_size: int = 20,
        min_history: int = 8,
        ema_alpha: float = 0.35,
        warmup_risk: float = 0.5,
        aggregation: str = "top2_mean",
        phase_low: float = 0.33,
        phase_high: float = 0.67,

        warmup_fraction: float = 1.0,
        exposure_start_cycle: float = 1.0,
        exposure_full_cycle: float = 2.5,

        motion_weight: float = 0.0,
        precision_weight: float = 1.0,
        exposure_weight: float = 1.0,
    ):
        self.window_size = int(window_size)
        self.min_history = int(min_history)
        self.ema_alpha = float(ema_alpha)
        self.warmup_risk = float(warmup_risk)
        self.aggregation = str(aggregation)
        self.phase_low = float(phase_low)
        self.phase_high = float(phase_high)

        self.warmup_fraction = float(warmup_fraction)
        self.exposure_start_cycle = float(exposure_start_cycle)
        self.exposure_full_cycle = float(exposure_full_cycle)

        self.motion_weight = float(motion_weight)
        self.precision_weight = float(precision_weight)
        self.exposure_weight = float(exposure_weight)

        self.histories = {
            name: deque(maxlen=self.window_size)
            for name in ONLINE_ADAPTIVE_SIGNAL_NAMES
        }

        self.risk_ema = None

    def _compute_exposure_score(self, exposure: dict):
        if exposure is None:
            return 0.0, {}

        stale_thr = max(1.0, float(exposure.get("stale_force_threshold", 8)))
        past_llm_calls = float(exposure.get("past_llm_calls", 0.0))

        exposure_start_calls = stale_thr * self.exposure_start_cycle
        exposure_full_calls = stale_thr * self.exposure_full_cycle

        call_exposure = np.clip(
            (past_llm_calls - exposure_start_calls)
            / max(1.0, exposure_full_calls - exposure_start_calls),
            0.0,
            1.0,
        )

        fixed_cache_age_mean = float(exposure.get("fixed_cache_age_mean", 0.0))
        wrist_cache_age_mean = float(exposure.get("wrist_cache_age_mean", 0.0))
        fixed_cache_age_max = float(exposure.get("fixed_cache_age_max", 0.0))
        wrist_cache_age_max = float(exposure.get("wrist_cache_age_max", 0.0))

        fixed_stale_ge_ratio = float(exposure.get("fixed_stale_ge_ratio", 0.0))
        wrist_stale_ge_ratio = float(exposure.get("wrist_stale_ge_ratio", 0.0))

        fixed_recency_blocked_ratio = float(exposure.get("fixed_recency_blocked_ratio", 0.0))
        wrist_recency_blocked_ratio = float(exposure.get("wrist_recency_blocked_ratio", 0.0))

        max_age_exposure = max(fixed_cache_age_max, wrist_cache_age_max) / stale_thr
        mean_age_exposure = max(fixed_cache_age_mean, wrist_cache_age_mean) / stale_thr
        stale_ge_exposure = max(fixed_stale_ge_ratio, wrist_stale_ge_ratio) / 100.0
        recency_block_exposure = max(fixed_recency_blocked_ratio, wrist_recency_blocked_ratio) / 100.0

        components = {
            "call_exposure": _clamp01(call_exposure),
            "max_age_exposure": _clamp01(max_age_exposure),
            "mean_age_exposure": _clamp01(mean_age_exposure),
            "stale_ge_exposure": _clamp01(stale_ge_exposure),
            "recency_block_exposure": _clamp01(recency_block_exposure),
            "exposure_start_calls": float(exposure_start_calls),
            "exposure_full_calls": float(exposure_full_calls),
        }

        age_exposure = max(
            0.40 * max_age_exposure,
            0.50 * mean_age_exposure,
        )

        # stale patch가 실제로 넓게 퍼졌을 때만 max_age를 강하게 믿음
        if stale_ge_exposure > 0.05 or recency_block_exposure > 0.05:
            age_exposure = max(age_exposure, 0.70 * max_age_exposure)

        exposure_score = max(
            call_exposure,
            age_exposure,
            stale_ge_exposure,
            recency_block_exposure,
        )

        return _clamp01(exposure_score), components

    def _compute_warmup_factor(self, exposure: dict):
        """
        초반에는 exposure가 낮아도 aggressive 금지.
        warmup_factor=0 -> cold-start
        warmup_factor=1 -> warmup 완료
        """
        if exposure is None:
            return 0.0

        stale_thr = max(1.0, float(exposure.get("stale_force_threshold", 8)))
        past_llm_calls = float(exposure.get("past_llm_calls", 0.0))

        warmup_total_calls = stale_thr * self.warmup_fraction

        return _clamp01(
            past_llm_calls / max(1.0, warmup_total_calls)
        )

    def score(self, signals: dict, exposure: dict = None):
        cleaned_signals = {}
        per_signal_rank = {}
        history_lengths = {}

        for name in ONLINE_ADAPTIVE_SIGNAL_NAMES:
            value = _valid_float_or_none(signals.get(name, None))
            cleaned_signals[name] = value

            hist = self.histories[name]
            history_lengths[name] = len(hist)

            if len(hist) < self.min_history:
                per_signal_rank[name] = self.warmup_risk
            else:
                per_signal_rank[name] = _percentile_rank(value, hist)

        # ------------------------------------------------------------
        # Motion score
        # ------------------------------------------------------------
        # motion은 위험도 그 자체가 아니라 frame-to-frame change의 크기.
        # pruning risk에는 낮은 weight로만 반영.
        motion_score = _aggregate_ranks(
            [per_signal_rank.get(name, 0.0) for name in MOTION_SIGNAL_NAMES],
            mode=self.aggregation,
        )

        # ------------------------------------------------------------
        # Speed interpretation
        # ------------------------------------------------------------
        # eef_speed high = transit 가능성.
        # eef_speed low + gripper/exposure = precision/contact risk 가능성.
        eef_speed_rank = float(per_signal_rank.get("recent_eef_speed", self.warmup_risk))
        low_speed_score = 1.0 - eef_speed_rank

        gripper_rank = float(per_signal_rank.get("recent_dgripper_asym", 0.0))

        # ------------------------------------------------------------
        # Exposure brake
        # ------------------------------------------------------------
        exposure_score, exposure_components = self._compute_exposure_score(exposure)

        warmup_factor = self._compute_warmup_factor(exposure)
        cold_start_brake = 1.0 - warmup_factor

        # ------------------------------------------------------------
        # Precision/contact risk
        # ------------------------------------------------------------
        # low speed alone is not risk.
        # low speed + gripper transition or accumulated exposure is risk.
        transition_score = _aggregate_ranks(
            [per_signal_rank.get(name, 0.0) for name in TRANSITION_SIGNAL_NAMES],
            mode=self.aggregation,
        )

        contact_score = _aggregate_ranks(
            [per_signal_rank.get(name, 0.0) for name in CONTACT_SIGNAL_NAMES],
            mode="max",
        )

        eef_speed_rank = float(per_signal_rank.get("recent_eef_speed", self.warmup_risk))
        low_speed_score = 1.0 - eef_speed_rank

        precision_score = max(
            transition_score * max(contact_score, 0.5 * exposure_score),
            low_speed_score * max(contact_score, 0.5 * exposure_score),

            # transition 자체가 매우 강하면 precision risk로 인정
            0.60 * transition_score,
        )

        # ------------------------------------------------------------
        # Final budget risk
        # ------------------------------------------------------------
        # risk가 높을수록 conservative target으로 이동.
        # motion은 낮은 weight.
        instant_risk = max(
            cold_start_brake,
            self.exposure_weight * exposure_score,
            self.precision_weight * precision_score,
            self.motion_weight * motion_score,
        )

        instant_risk = _clamp01(instant_risk)

        if self.risk_ema is None:
            self.risk_ema = float(instant_risk)
        else:
            self.risk_ema = (
                self.ema_alpha * float(instant_risk)
                + (1.0 - self.ema_alpha) * float(self.risk_ema)
            )

        risk_score = _clamp01(self.risk_ema)

        if risk_score >= self.phase_high:
            global_phase = "risky"        # compatibility label
            budget_mode = "conservative"
        elif risk_score >= self.phase_low:
            global_phase = "normal"
            budget_mode = "balanced"
        else:
            global_phase = "stable_late"  # compatibility label
            budget_mode = "aggressive"

        score_components = {
            "motion_score": float(motion_score),
            "exposure_score": float(exposure_score),
            "precision_score": float(precision_score),
            "cold_start_brake": float(cold_start_brake),
            "warmup_factor": float(warmup_factor),
            "eef_speed_rank": float(eef_speed_rank),
            "low_speed_score": float(low_speed_score),
            "gripper_rank": float(gripper_rank),
        }

        reasons = [
            f"mode={budget_mode}",
            f"motion={motion_score:.2f}",
            f"exposure={exposure_score:.2f}",
            f"precision={precision_score:.2f}",
            f"cold_start={cold_start_brake:.2f}",
            f"eef_speed_rank={eef_speed_rank:.2f}",
            f"low_speed={low_speed_score:.2f}",
            f"gripper_rank={gripper_rank:.2f}",
        ]

        phase_info = {
            "global_phase": global_phase,
            "budget_mode": budget_mode,

            "risk_score": risk_score,
            "instant_risk_score": float(instant_risk),

            "per_signal_risk": dict(per_signal_rank),  # debug 호환용 이름
            "per_signal_rank": dict(per_signal_rank),

            "score_components": score_components,
            "exposure_components": exposure_components,

            "dominant_signal": max(
                score_components.items(),
                key=lambda kv: kv[1],
            )[0],

            "reasons": reasons,
            "fixed_reasons": reasons,
            "wrist_reasons": reasons,

            "signals": cleaned_signals,
            "exposure": exposure if exposure is not None else {},

            "history_lengths": dict(history_lengths),
            "aggregation": self.aggregation,
            "window_size": self.window_size,
            "min_history": self.min_history,
            "ema_alpha": self.ema_alpha,
        }

        # 현재 값을 scoring한 뒤 history에 넣는다.
        for name, value in cleaned_signals.items():
            if value is not None:
                self.histories[name].append(float(value))

        return phase_info


def build_online_global_adaptive_pruning_plan(
    risk_controller: OnlineAdaptiveRiskController,
    signals: dict,
    n_candidates: int,
    ref_total_tokens: int,
    fixed_available: int = 0,
    wrist_available: int = 0,
    exposure: dict = None,
    min_total_prune_ratios=(0.078, 0.235, 0.313),
    max_total_prune_ratios=(0.158, 0.290, 0.485),
):
    """
    Online controller로 target ratio와 candidate-list ratio까지 계산.
    run_libero_eval.py에서는 이 함수만 호출하면 됨.
    """

    adaptive_phase_info = risk_controller.score(
        signals=signals,
        exposure=exposure,
    )

    target_total_ratios = interpolate_total_prune_ratios_from_risk(
        risk_score=adaptive_phase_info["risk_score"],
        min_total_prune_ratios=min_total_prune_ratios,
        max_total_prune_ratios=max_total_prune_ratios,
    )

    dynamic_ratios, target_counts = total_ratio_targets_to_candidate_ratios(
        target_total_ratios=target_total_ratios,
        n_candidates=n_candidates,
        ref_total_tokens=ref_total_tokens,
    )

    stats = {
        "global_phase": adaptive_phase_info["global_phase"],
        "budget_mode": adaptive_phase_info["budget_mode"],

        "risk_score": adaptive_phase_info["risk_score"],
        "instant_risk_score": adaptive_phase_info["instant_risk_score"],

        "per_signal_risk": adaptive_phase_info["per_signal_risk"],
        "per_signal_rank": adaptive_phase_info["per_signal_rank"],

        "score_components": adaptive_phase_info["score_components"],
        "exposure_components": adaptive_phase_info["exposure_components"],

        "dominant_signal": adaptive_phase_info["dominant_signal"],
        "reasons": adaptive_phase_info["reasons"],
        "fixed_reasons": adaptive_phase_info["fixed_reasons"],
        "wrist_reasons": adaptive_phase_info["wrist_reasons"],

        "target_total_ratios": target_total_ratios,
        "target_counts": target_counts,
        "dynamic_ratios": dynamic_ratios,

        "num_candidates": int(n_candidates),
        "ref_total_tokens": int(ref_total_tokens),
        "fixed_available": int(fixed_available),
        "wrist_available": int(wrist_available),

        "history_lengths": adaptive_phase_info["history_lengths"],
        "aggregation": adaptive_phase_info["aggregation"],
        "exposure": exposure if exposure is not None else {},
    }

    return (
        adaptive_phase_info,
        target_total_ratios,
        dynamic_ratios,
        target_counts,
        stats,
    )


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
