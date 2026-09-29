"""
kinematic_budget_utils.py

kinematic budget signal for LIBERO/OpenVLA-OFT.

The current policy query cannot use its yet-uncomputed action chunk.  We
therefore analyze the previously generated/executed action chunk and assume
that its terminal motion regime persists into the next query.

This module contains NO image projection, NO ROI patch generation, and NO
spatial patch protection.  Kinematics changes only the progressive pruning
budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ApproachSignal:
    # Compatibility name: ``active`` now means interaction-sensitive / fine.
    active: bool
    reason: str
    slowdown_score: float
    valley_lead_steps: int   # compatibility: valley index inside previous chunk
    rotation_stabilization_score: float
    gripper_lead_steps: int  # compatibility: transition index inside previous chunk
    target_gripper_state: Optional[str]
    preview_len: int

    # Explicit terminal-semantics diagnostics.
    regime: str = "free_motion"
    slowdown_source: str = "none"
    valley_age_steps: int = -1
    gripper_age_steps: int = -1

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _translation_norm(action_vec: np.ndarray) -> float:
    """OpenVLA robot-facing action uses xyz in the first three dims."""
    arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return 0.0
    return float(np.linalg.norm(arr[:3]))


def _rotation_norm(action_vec: np.ndarray) -> float:
    arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if arr.size < 6:
        return 0.0
    return float(np.linalg.norm(arr[3:6]))


def _smooth_profile(profile: Sequence[float], window: int = 3) -> List[float]:
    values = [float(x) for x in profile]
    if window <= 1 or len(values) < window:
        return values
    half = window // 2
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


def _find_valleys_with_prominence(profile: Sequence[float]) -> List[Tuple[int, float]]:
    values = [float(x) for x in profile]
    valleys: List[Tuple[int, float]] = []
    for i in range(1, len(values) - 1):
        if (
            values[i] <= values[i - 1]
            and values[i] <= values[i + 1]
            and (values[i] < values[i - 1] or values[i] < values[i + 1])
        ):
            left_rise = max(values[: i + 1]) - values[i]
            right_rise = max(values[i:]) - values[i]
            valleys.append((i, float(min(left_rise, right_rise))))
    return valleys


def _gripper_state(value: float, threshold: float = 0.5) -> str:
    # Existing LIBERO/OpenVLA convention used by this evaluator.
    return "open" if float(value) >= float(threshold) else "closed"


def _latest_gripper_transition(
    chunk: Sequence[np.ndarray],
    *,
    threshold: float,
) -> Tuple[int, Optional[str]]:
    if not chunk:
        return -1, None
    first = np.asarray(chunk[0], dtype=np.float32).reshape(-1)
    if first.size == 0:
        return -1, None
    prev = _gripper_state(float(first[-1]), threshold)
    latest_idx = -1
    latest_target: Optional[str] = None
    for i in range(1, len(chunk)):
        arr = np.asarray(chunk[i], dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        cur = _gripper_state(float(arr[-1]), threshold)
        if cur != prev:
            latest_idx = int(i)
            latest_target = cur
        prev = cur
    return latest_idx, latest_target


def detect_approach_signal(
    queued_raw_actions: Sequence[np.ndarray],
    *,
    preview_steps: int = 8,
    slowdown_threshold: float = 0.18,
    smoothing_window: int = 3,
    lead_steps_threshold: int = 4,
    gripper_action_threshold: float = 0.5,
    close_action_positive: bool = True,
) -> ApproachSignal:
    """Classify the terminal regime of the *previous* action chunk.

    FREE MOTION (alpha=1): coherent/accelerating motion with no recent
    interaction cue.

    FINE MANIPULATION / interaction-sensitive (alpha=0):
      - a sufficiently strong slowdown that persists toward the chunk end, or
      - a gripper transition occurring within the final terminal window.

    Valley handling:
      1) Find local valleys using normalized prominence.
      2) Only a valley near the END of the previous chunk can describe the
         terminal regime.  ``age = H-1-index`` must be <= terminal window.
      3) If no terminal valley exists, use endpoint decline as the monotonic
         deceleration fallback.  Monotonic acceleration clips to zero.

    ``lead_steps_threshold`` is retained for config compatibility, but for a
    previous/executed chunk it is interpreted as a TERMINAL-WINDOW size, not
    as future look-ahead.
    """
    # We care about the most recently executed part of the previous chunk.
    preview = list(queued_raw_actions)[-int(preview_steps):]
    n = len(preview)
    if n < 3:
        return ApproachSignal(
            active=False,
            reason="insufficient_history",
            slowdown_score=0.0,
            valley_lead_steps=-1,
            rotation_stabilization_score=0.0,
            gripper_lead_steps=-1,
            target_gripper_state=None,
            preview_len=n,
            regime="free_motion",
            slowdown_source="none",
            valley_age_steps=-1,
            gripper_age_steps=-1,
        )

    translations = [_translation_norm(a) for a in preview]
    rotations = [_rotation_norm(a) for a in preview]
    smoothed = _smooth_profile(translations, window=int(smoothing_window))

    terminal_window = max(0, int(lead_steps_threshold))
    valleys = _find_valleys_with_prominence(smoothed)

    # Prefer valleys that describe the terminal region, not an early event
    # from which motion has already recovered.
    terminal_valleys: List[Tuple[int, float, int]] = []
    for idx, prominence in valleys:
        age = (n - 1) - int(idx)
        if age <= terminal_window:
            terminal_valleys.append((int(idx), float(prominence), int(age)))

    slowdown_score = 0.0
    slowdown_source = "none"
    valley_idx = -1
    valley_age = -1

    if terminal_valleys:
        valley_idx, prominence, valley_age = max(
            terminal_valleys, key=lambda x: x[1]
        )
        denom = max(max(smoothed), 1e-6)
        slowdown_score = float(np.clip(prominence / denom, 0.0, 1.0))
        slowdown_source = "terminal_valley"
    else:
        # No usable valley: sustained deceleration can end at the window
        # boundary and therefore never form an interior local minimum.
        # Endpoint decline preserves that signal; acceleration yields 0.
        first_val = float(smoothed[0])
        last_val = float(smoothed[-1])
        slowdown_score = float(
            np.clip((first_val - last_val) / (abs(first_val) + 1e-6), 0.0, 1.0)
        )
        if slowdown_score > 0.0:
            slowdown_source = "endpoint_deceleration_fallback"
            valley_idx = n - 1
            valley_age = 0

    # Keep rotation statistic only as a diagnostic; it does not control A4.
    smoothed_rot = _smooth_profile(rotations, window=int(smoothing_window))
    rot_valleys = _find_valleys_with_prominence(smoothed_rot)
    if rot_valleys:
        _, rot_prom = max(rot_valleys, key=lambda x: x[1])
        rotation_score = float(
            np.clip(rot_prom / (max(max(smoothed_rot), 1e-6)), 0.0, 1.0)
        )
    else:
        rotation_score = 0.0

    grip_idx, target_gripper_state = _latest_gripper_transition(
        preview, threshold=float(gripper_action_threshold)
    )
    grip_age = (n - 1 - grip_idx) if grip_idx >= 0 else -1
    recent_gripper = grip_idx >= 0 and grip_age <= terminal_window
    slowdown_active = slowdown_score >= float(slowdown_threshold)

    # Binary final-paper scheduler.  This intentionally folds the old
    # approach/pre-grasp/pre-place internal labels into one conservative
    # interaction-sensitive operating point.
    fine = bool(slowdown_active or recent_gripper)
    reasons: List[str] = []
    if slowdown_active:
        reasons.append(slowdown_source)
    if recent_gripper:
        if target_gripper_state == "closed":
            reasons.append("pre_grasp")
        elif target_gripper_state == "open":
            reasons.append("pre_place")
        else:
            reasons.append("gripper_transition")

    return ApproachSignal(
        active=fine,
        reason="+".join(reasons) if reasons else "coherent_motion",
        slowdown_score=float(slowdown_score),
        valley_lead_steps=int(valley_idx),
        rotation_stabilization_score=float(rotation_score),
        gripper_lead_steps=int(grip_idx),
        target_gripper_state=target_gripper_state,
        preview_len=n,
        regime="fine_manipulation" if fine else "free_motion",
        slowdown_source=slowdown_source,
        valley_age_steps=int(valley_age),
        gripper_age_steps=int(grip_age),
    )
