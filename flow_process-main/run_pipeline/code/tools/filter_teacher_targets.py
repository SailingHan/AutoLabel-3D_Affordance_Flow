#!/usr/bin/env python3
"""Compute automatic quality flags for exported TraceForge teacher targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def finite_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).sum() / arr.size)


def finite_ratio_visible(arr: np.ndarray, visibility: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    finite = np.isfinite(arr).all(axis=-1)
    visible = visibility.astype(bool)
    if visible.shape != finite.shape:
        return finite_ratio(arr)
    if not visible.any():
        return 0.0
    return float((finite & visible).sum() / visible.sum())


def mean_motion(trace: np.ndarray, visibility: np.ndarray) -> float:
    if trace.ndim != 3 or trace.shape[1] < 2:
        return 0.0
    finite = np.isfinite(trace).all(axis=-1)
    visible = visibility.astype(bool) & finite
    pair_visible = visible[:, 1:] & visible[:, :-1]
    if not pair_visible.any():
        return 0.0
    delta = np.linalg.norm(np.diff(np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0), axis=1), axis=-1)
    return float(delta[pair_visible].mean()) if pair_visible.any() else 0.0


def flow_count(episode_dir: Path, name: str) -> int:
    path = episode_dir / name
    if not path.exists():
        return 0
    try:
        arr = np.load(path)
    except Exception:
        return 0
    return int(arr.shape[0]) if arr.ndim > 0 else 0


def has_fallback_marker(episode_dir: Path) -> bool:
    for path in episode_dir.rglob("*.json"):
        try:
            text = path.read_text(encoding="utf-8").lower()
        except Exception:
            continue
        if "fallback" in text and ("grid" in text or "mask" in text):
            return True
    return False


def score_episode(
    episode_dir: Path,
    *,
    min_queries: int,
    min_valid_steps: int,
    max_nonfinite_ratio: float,
    static_motion_eps: float,
    require_hand_flow: bool,
    require_target_flow: bool,
) -> dict[str, Any]:
    target_path = episode_dir / "teacher_targets.npz"
    flags: list[str] = []
    if not target_path.exists():
        return {
            "episode_dir": str(episode_dir),
            "quality_score": 0.0,
            "quality_flags": ["missing_teacher_targets"],
            "is_usable_teacher": False,
        }

    with np.load(target_path) as data:
        trace_3d = np.asarray(data["trace_3d"], dtype=np.float32)
        trace_2d = np.asarray(data["trace_2d"], dtype=np.float32)
        valid_steps = np.asarray(data["valid_steps"]).astype(bool)
        visibility = np.asarray(data["visibility"]).astype(bool)
        role_id = np.asarray(data["role_id"], dtype=np.int32)

    n_queries = int(trace_3d.shape[0]) if trace_3d.ndim >= 1 else 0
    if n_queries < min_queries:
        flags.append("query_count_too_low")

    if valid_steps.ndim == 2:
        per_track_valid = valid_steps.sum(axis=1)
        max_valid = int(per_track_valid.max()) if per_track_valid.size else 0
    else:
        max_valid = int(valid_steps.sum()) if valid_steps.size else 0
    if max_valid < min_valid_steps:
        flags.append("valid_steps_too_short")

    nonfinite_ratio = 1.0 - min(
        finite_ratio_visible(trace_3d, visibility),
        finite_ratio_visible(trace_2d, visibility),
    )
    if nonfinite_ratio > max_nonfinite_ratio:
        flags.append("nonfinite_ratio_too_high")

    motion = mean_motion(trace_3d, visibility)
    if motion <= static_motion_eps:
        flags.append("all_static_coords")

    hand_count = flow_count(episode_dir, "hand_flow.npy")
    target_count = flow_count(episode_dir, "target_flow.npy")
    if require_hand_flow and hand_count == 0:
        flags.append("empty_hand_flow")
    if require_target_flow and target_count == 0:
        flags.append("empty_target_flow")

    if has_fallback_marker(episode_dir):
        flags.append("mask_fallback_detected")

    role_nonzero = int((role_id > 0).sum()) if role_id.size else 0
    if role_nonzero == 0:
        flags.append("missing_role_ids")

    penalties = {
        "query_count_too_low": 0.25,
        "valid_steps_too_short": 0.25,
        "nonfinite_ratio_too_high": 0.30,
        "all_static_coords": 0.35,
        "empty_hand_flow": 0.20,
        "empty_target_flow": 0.30,
        "mask_fallback_detected": 1.00,
        "missing_role_ids": 0.25,
    }
    quality_score = max(0.0, 1.0 - sum(penalties.get(flag, 0.1) for flag in set(flags)))
    return {
        "episode_dir": str(episode_dir),
        "quality_score": float(quality_score),
        "quality_flags": flags,
        "is_usable_teacher": len(flags) == 0,
        "metrics": {
            "num_queries": n_queries,
            "max_valid_steps": max_valid,
            "nonfinite_ratio": float(nonfinite_ratio),
            "mean_motion": float(motion),
            "hand_flow_count": hand_count,
            "target_flow_count": target_count,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser("Filter exported teacher_targets.npz assets")
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--min-queries", type=int, default=16)
    parser.add_argument("--min-valid-steps", type=int, default=4)
    parser.add_argument("--max-nonfinite-ratio", type=float, default=0.25)
    parser.add_argument("--static-motion-eps", type=float, default=1e-5)
    parser.add_argument("--allow-missing-hand-flow", action="store_true")
    parser.add_argument("--allow-missing-target-flow", action="store_true")
    args = parser.parse_args()

    rows = []
    for episode_dir in sorted(p for p in args.teacher_output.iterdir() if p.is_dir()):
        rows.append(
            score_episode(
                episode_dir,
                min_queries=args.min_queries,
                min_valid_steps=args.min_valid_steps,
                max_nonfinite_ratio=args.max_nonfinite_ratio,
                static_motion_eps=args.static_motion_eps,
                require_hand_flow=not args.allow_missing_hand_flow,
                require_target_flow=not args.allow_missing_target_flow,
            )
        )

    usable = [row for row in rows if row["is_usable_teacher"]]
    report = {
        "teacher_output": str(args.teacher_output),
        "num_episodes": len(rows),
        "num_usable": len(usable),
        "episodes": rows,
    }
    out_path = args.teacher_output / "teacher_quality.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[teacher_filter] usable={len(usable)}/{len(rows)} report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
