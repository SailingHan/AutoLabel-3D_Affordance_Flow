#!/usr/bin/env python3
"""Export stable training assets from TraceForge episode outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


KEEP_FILES = (
    "selected_traces.npy",
    "hand_flow.npy",
    "tool_flow.npy",
    "target_flow.npy",
    "query_role_id.npy",
    "query_role_name.json",
    "query_frame_index.npy",
)


def build_visibility(valid_steps: np.ndarray, n_tracks: int, traj_len: int) -> np.ndarray:
    valid = np.asarray(valid_steps).astype(bool).reshape(-1)
    visibility = np.zeros((n_tracks, traj_len), dtype=bool)
    use_len = min(traj_len, valid.shape[0])
    if use_len > 0:
        visibility[:, :use_len] = valid[:use_len][None, :]
    return visibility


def align_time_length(arr: np.ndarray, target_len: int, fill_value: float = -np.inf) -> np.ndarray:
    if arr.shape[1] == target_len:
        return arr
    out = np.full((arr.shape[0], target_len, arr.shape[2]), fill_value, dtype=arr.dtype)
    use_len = min(target_len, arr.shape[1])
    if use_len > 0:
        out[:, :use_len, :] = arr[:, :use_len, :]
    return out


def load_role_ids(traceforge_episode_dir: Path, sample_stems: list[str]) -> np.ndarray:
    role_chunks = []
    role_dir = traceforge_episode_dir / "role_queries"
    for stem in sample_stems:
        role_path = role_dir / f"{stem}_role_id.npy"
        if role_path.exists():
            role_chunks.append(np.asarray(np.load(role_path), dtype=np.int32).reshape(-1))
    if role_chunks:
        return np.concatenate(role_chunks, axis=0).astype(np.int32)
    role_path = traceforge_episode_dir / "query_role_id.npy"
    if role_path.exists():
        return np.asarray(np.load(role_path), dtype=np.int32).reshape(-1)
    return np.zeros((0,), dtype=np.int32)


def export_episode(traceforge_episode_dir: Path, output_episode_dir: Path, overwrite: bool = False) -> Path | None:
    samples_dir = traceforge_episode_dir / "samples"
    if not samples_dir.exists():
        return None
    output_episode_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_episode_dir / "teacher_targets.npz"
    if target_path.exists() and not overwrite:
        return target_path

    trace_3d_chunks = []
    trace_2d_chunks = []
    valid_chunks = []
    visibility_chunks = []
    query_frame_chunks = []
    sample_stems = []

    for sample_path in sorted(samples_dir.glob("*.npz")):
        with np.load(sample_path, allow_pickle=True) as sample:
            if "traj" not in sample or "traj_2d" not in sample:
                continue
            trace_3d = np.asarray(sample["traj"], dtype=np.float32)
            trace_2d = np.asarray(sample["traj_2d"], dtype=np.float32)
            if trace_3d.ndim != 3 or trace_2d.ndim != 3:
                continue
            n_tracks, traj_len = trace_3d.shape[:2]
            trace_2d = align_time_length(trace_2d, target_len=traj_len)
            valid_steps = np.asarray(
                sample["valid_steps"] if "valid_steps" in sample.files else np.ones((traj_len,), dtype=bool)
            ).astype(bool)
            frame_index = int(
                np.asarray(sample["frame_index"] if "frame_index" in sample.files else np.array([0])).reshape(-1)[0]
            )

        trace_3d_chunks.append(trace_3d)
        trace_2d_chunks.append(trace_2d)
        valid_chunks.append(np.tile(valid_steps.reshape(1, -1), (n_tracks, 1)))
        visibility_chunks.append(build_visibility(valid_steps, n_tracks=n_tracks, traj_len=traj_len))
        query_frame_chunks.append(np.full((n_tracks,), frame_index, dtype=np.int32))
        sample_stems.append(sample_path.stem)

    if not trace_3d_chunks:
        return None

    trace_3d = np.concatenate(trace_3d_chunks, axis=0).astype(np.float32)
    trace_2d = np.concatenate(trace_2d_chunks, axis=0).astype(np.float32)
    valid_steps = np.concatenate(valid_chunks, axis=0).astype(bool)
    visibility = np.concatenate(visibility_chunks, axis=0).astype(bool)
    query_frame_index = np.concatenate(query_frame_chunks, axis=0).astype(np.int32)
    role_id = load_role_ids(traceforge_episode_dir, sample_stems)
    if role_id.shape[0] != trace_3d.shape[0]:
        fallback_role = np.zeros((trace_3d.shape[0],), dtype=np.int32)
        use = min(role_id.shape[0], fallback_role.shape[0])
        fallback_role[:use] = role_id[:use]
        role_id = fallback_role

    np.savez_compressed(
        target_path,
        trace_3d=trace_3d,
        trace_2d=trace_2d,
        valid_steps=valid_steps,
        visibility=visibility,
        query_frame_index=query_frame_index,
        role_id=role_id.astype(np.int32),
    )

    for name in KEEP_FILES:
        src = traceforge_episode_dir / name
        if src.exists():
            shutil.copy2(src, output_episode_dir / name)
    return target_path


def main() -> int:
    parser = argparse.ArgumentParser("Export teacher_targets.npz from TraceForge outputs")
    parser.add_argument("--traceforge-output", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.teacher_output.mkdir(parents=True, exist_ok=True)
    exported = []
    for episode_dir in sorted(p for p in args.traceforge_output.iterdir() if p.is_dir()):
        out_dir = args.teacher_output / episode_dir.name
        target = export_episode(episode_dir, out_dir, overwrite=args.overwrite)
        if target is not None:
            exported.append(str(target))

    manifest = {"traceforge_output": str(args.traceforge_output), "teacher_targets": exported}
    with open(args.teacher_output / "teacher_targets_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[teacher_export] exported {len(exported)} teacher target files to {args.teacher_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
