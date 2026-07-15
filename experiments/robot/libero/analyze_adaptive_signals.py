import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit("Usage: python analyze_adaptive_signals_by_episode.py <log_file.txt>")

log_path = Path(sys.argv[1])

signal_keys = [
    "hole_ratio",
    "cam_trans_delta",
    "cam_rot_delta_rad",
    "recent_eef_speed",
    "recent_dgripper_asym",
]

rows = []
episode_meta = {}

episode_id = 0
current_task = "NA"

def parse_float_token(v):
    if v is None or v == "NA":
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan

with log_path.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()

        # 보통 각 rollout 시작 때 Task가 찍히면 episode 시작으로 봄
        if line.startswith("Task:") or " Task:" in line:
            episode_id += 1
            current_task = line.split("Task:", 1)[-1].strip()
            episode_meta[episode_id] = {
                "episode_id": episode_id,
                "task": current_task,
                "success": np.nan,
            }
            continue

        # Success line이 episode 끝에 찍히는 경우
        if "Success:" in line and episode_id > 0:
            m = re.search(r"Success:\s*(True|False)", line)
            if m:
                episode_meta.setdefault(episode_id, {
                    "episode_id": episode_id,
                    "task": current_task,
                    "success": np.nan,
                })
                episode_meta[episode_id]["success"] = (m.group(1) == "True")
            continue

        if "[ADAPTIVE PRUNING V2]" not in line and "[ADAPTIVE PRUNING ONLINE]" not in line:
            continue

        row = {
            "episode_id": episode_id,
            "task": current_task,
        }

        m = re.search(r"step=(\d+)", line)
        row["step"] = int(m.group(1)) if m else np.nan

        m = re.search(r"llm_call=(\d+)", line)
        row["llm_call"] = int(m.group(1)) if m else np.nan

        m = re.search(r"global_phase=([A-Za-z_]+)", line)
        row["global_phase"] = m.group(1) if m else "NA"

        m = re.search(r"risk_score=([0-9.]+)", line)
        row["risk_score"] = float(m.group(1)) if m else np.nan

        for k in signal_keys:
            m = re.search(rf"{k}=([^ ]+)", line)
            row[k] = parse_float_token(m.group(1)) if m else np.nan

        # target_counts=(48, 120, 186) 형태도 저장
        m = re.search(r"target_counts=\(([^)]*)\)", line)
        if m:
            counts = [x.strip() for x in m.group(1).split(",")]
            for i, name in enumerate(["target_L2", "target_L6", "target_L10"]):
                try:
                    row[name] = int(counts[i])
                except Exception:
                    row[name] = np.nan
        else:
            row["target_L2"] = np.nan
            row["target_L6"] = np.nan
            row["target_L10"] = np.nan

        rows.append(row)

df = pd.DataFrame(rows)

if len(df) == 0:
    raise SystemExit("No adaptive pruning lines found.")

meta_df = pd.DataFrame(list(episode_meta.values()))
df = df.merge(meta_df[["episode_id", "success"]], on="episode_id", how="left")

# ------------------------------------------------------------
# 전체 call csv
# ------------------------------------------------------------
all_csv = log_path.with_name(log_path.stem + "_adaptive_signals_all.csv")
df.to_csv(all_csv, index=False)

# ------------------------------------------------------------
# episode별 summary
# ------------------------------------------------------------
summary_rows = []

for ep, g in df.groupby("episode_id"):
    out = {
        "episode_id": ep,
        "task": g["task"].iloc[0],
        "success": g["success"].iloc[0],
        "n_calls": len(g),
    }

    phase_counts = g["global_phase"].value_counts()
    for phase in ["stable_late", "normal", "risky"]:
        out[f"phase_{phase}_count"] = int(phase_counts.get(phase, 0))
        out[f"phase_{phase}_ratio"] = float(phase_counts.get(phase, 0) / max(1, len(g)))

    for k in signal_keys:
        vals = g[k].dropna().to_numpy()

        if len(vals) == 0:
            for p in [50, 75, 80, 90, 95, 99]:
                out[f"{k}_p{p}"] = np.nan
            out[f"{k}_max"] = np.nan
            out[f"{k}_mean"] = np.nan
            continue

        for p in [50, 75, 80, 90, 95, 99]:
            out[f"{k}_p{p}"] = float(np.percentile(vals, p))

        out[f"{k}_max"] = float(np.max(vals))
        out[f"{k}_mean"] = float(np.mean(vals))

    for k in ["target_L2", "target_L6", "target_L10"]:
        vals = g[k].dropna().to_numpy()
        out[f"{k}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
        out[f"{k}_max"] = float(np.max(vals)) if len(vals) else np.nan

    summary_rows.append(out)

summary_df = pd.DataFrame(summary_rows)

summary_csv = log_path.with_name(log_path.stem + "_adaptive_episode_summary.csv")
summary_df.to_csv(summary_csv, index=False)

# ------------------------------------------------------------
# failed only
# ------------------------------------------------------------
failed_df = df[df["success"] == False].copy()
failed_csv = log_path.with_name(log_path.stem + "_adaptive_failed_only.csv")
failed_df.to_csv(failed_csv, index=False)

print(f"Saved all calls: {all_csv}")
print(f"Saved episode summary: {summary_csv}")
print(f"Saved failed only: {failed_csv}")
print()
print("Episode summary:")
print(summary_df[
    [
        "episode_id",
        "success",
        "n_calls",
        "phase_stable_late_count",
        "phase_stable_late_ratio",
        "phase_normal_count",
        "phase_risky_count",
        "recent_eef_speed_p80",
        "hole_ratio_p80",
        "cam_trans_delta_p80",
        "target_L10_mean",
    ]
].to_string(index=False))