#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

TRACEFORGE_ROOT = os.environ.get("TRACEFORGE_ROOT", "/home/zhy/data/TraceForge")
if TRACEFORGE_ROOT and TRACEFORGE_ROOT not in sys.path:
    sys.path.insert(0, TRACEFORGE_ROOT)

from third_party.cotracker.visualizer import Visualizer


ROLE_COLORS = {
    0: (240, 240, 240),
    1: (255, 64, 64),
    2: (64, 180, 255),
    3: (80, 230, 120),
    4: (255, 210, 64),
}


def load_sample(npz_path: Path) -> dict:
    data = np.load(npz_path)
    out = {k: data[k] for k in data.files}
    data.close()
    return out


def load_rgb_frames(images_dir: Path, video_name: str, start_frame: int, num_frames: int) -> np.ndarray:
    frames = []
    for frame_idx in range(start_frame, start_frame + num_frames):
        image_path = images_dir / f"{video_name}_{frame_idx}.png"
        if not image_path.exists():
            break
        frames.append(np.asarray(Image.open(image_path).convert("RGB")))
    if not frames:
        raise FileNotFoundError(
            f"No readable frames found under {images_dir} for {video_name} starting at frame {start_frame}"
        )
    return np.stack(frames, axis=0)


def build_visibility(n_tracks: int, traj_len: int, valid_steps: np.ndarray | None) -> np.ndarray:
    visibility = np.ones((traj_len, n_tracks), dtype=bool)
    if valid_steps is None:
        return visibility
    valid_steps = np.asarray(valid_steps).astype(bool).reshape(-1)
    use_len = min(traj_len, valid_steps.shape[0])
    visibility[use_len:] = False
    if use_len > 0:
        visibility[:use_len] = valid_steps[:use_len, None]
    return visibility


def sanitize_tracks(traj_2d: np.ndarray, visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(traj_2d).all(axis=-1)
    visibility = visibility & finite
    traj_2d = traj_2d.copy()
    traj_2d[~finite] = 0.0
    return traj_2d, visibility


def natural_frame_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_semantics(traj_dir: Path) -> dict:
    path = traj_dir / "llm_semantics.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("llm_result", {})


def make_adapter_previews(adapter_root: Path, out_dir: Path, max_frames: int = 4) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for traj_dir in sorted(p for p in adapter_root.glob("*/*") if p.is_dir()):
        frames = sorted(traj_dir.glob("rgb_*.png"), key=natural_frame_key)
        if not frames:
            continue
        if len(frames) > max_frames:
            picks = sorted({round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)})
            frames = [frames[int(i)] for i in picks]
        semantics = load_semantics(traj_dir)
        fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))
        if len(frames) == 1:
            axes = [axes]
        for ax, frame in zip(axes, frames):
            ax.imshow(Image.open(frame).convert("RGB"))
            ax.set_title(frame.name)
            ax.axis("off")
        fig.suptitle(
            f"{traj_dir.parent.name}/{traj_dir.name}\n"
            f"{semantics.get('tool', '?')} -> {semantics.get('target', '?')} | "
            f"{semantics.get('action', '?')} | {semantics.get('core_description', '')}"
        )
        fig.tight_layout()
        out_path = out_dir / f"{traj_dir.parent.name}__{traj_dir.name}_adapter_preview.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def role_ids_for_sample(episode_dir: Path, sample_stem: str, n_points: int) -> np.ndarray:
    role_path = episode_dir / "role_queries" / f"{sample_stem}_role_id.npy"
    if role_path.exists():
        role_ids = np.asarray(np.load(role_path), dtype=np.int32).reshape(-1)
        if role_ids.shape[0] == n_points:
            return role_ids
    return np.zeros((n_points,), dtype=np.int32)


def draw_query_overlay(image: Image.Image, keypoints: np.ndarray, role_ids: np.ndarray) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    step = max(1, keypoints.shape[0] // 500)
    for xy, role_id in zip(keypoints[::step], role_ids[::step]):
        x, y = float(xy[0]), float(xy[1])
        color = ROLE_COLORS.get(int(role_id), (255, 255, 255))
        draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=(*color, 210), outline=(0, 0, 0, 180))
    return out


def draw_trace_overlay(image: Image.Image, traj_2d: np.ndarray, role_ids: np.ndarray) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    step = max(1, traj_2d.shape[0] // 120)
    finite = np.isfinite(traj_2d).all(axis=-1)
    for track, ok, role_id in zip(traj_2d[::step], finite[::step], role_ids[::step]):
        pts = [(float(xy[0]), float(xy[1])) for xy, valid in zip(track, ok) if valid]
        if len(pts) < 2:
            continue
        color = ROLE_COLORS.get(int(role_id), (255, 255, 255))
        draw.line(pts, fill=(*color, 190), width=2)
        x0, y0 = pts[0]
        draw.ellipse((x0 - 3, y0 - 3, x0 + 3, y0 + 3), fill=(*color, 230))
    return out


def make_batch_overlays(batch_root: Path, out_dir: Path, max_samples_per_episode: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for episode_dir in sorted(p for p in batch_root.iterdir() if p.is_dir()):
        samples = sorted((episode_dir / "samples").glob("*.npz"))[:max_samples_per_episode]
        for sample_path in samples:
            sample = load_sample(sample_path)
            keypoints = np.asarray(sample["keypoints"], dtype=np.float32)
            traj_2d = np.asarray(sample["traj_2d"], dtype=np.float32)
            frame_index = int(np.asarray(sample["frame_index"]).reshape(-1)[0])
            image_path = episode_dir / "images" / f"{episode_dir.name}_{frame_index}.png"
            if not image_path.exists():
                continue
            role_ids = role_ids_for_sample(episode_dir, sample_path.stem, keypoints.shape[0])
            image = Image.open(image_path).convert("RGB")
            query_out = out_dir / f"{episode_dir.name}_{frame_index}_queries.png"
            draw_query_overlay(image, keypoints, role_ids).save(query_out)
            outputs.append(query_out)
            trace_out = out_dir / f"{episode_dir.name}_{frame_index}_trace2d.png"
            draw_trace_overlay(image, traj_2d, role_ids).save(trace_out)
            outputs.append(trace_out)
    return outputs


def make_flow_summaries(batch_root: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for episode_dir in sorted(p for p in batch_root.iterdir() if p.is_dir()):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection="3d")
        any_flow = False
        for name, color in [("hand_flow.npy", "red"), ("tool_flow.npy", "deepskyblue"), ("target_flow.npy", "limegreen")]:
            path = episode_dir / name
            if not path.exists():
                continue
            arr = np.asarray(np.load(path), dtype=np.float32)
            if arr.ndim != 3 or arr.shape[0] == 0:
                continue
            any_flow = True
            for track in arr[: min(30, arr.shape[0])]:
                finite = np.isfinite(track).all(axis=-1)
                pts = track[finite]
                if pts.shape[0] >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.35, linewidth=1)
            ax.plot([], [], [], color=color, label=f"{name[:-4]} n={arr.shape[0]}")
        if not any_flow:
            plt.close(fig)
            continue
        ax.set_title(episode_dir.name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(loc="best")
        out_path = out_dir / f"{episode_dir.name}_flow3d_summary.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def main():
    ap = argparse.ArgumentParser("Render one TraceForge sample's 2D flow as an overlay MP4")
    ap.add_argument("--sample-npz", type=Path, default=None, help="Path to one samples/*.npz file")
    ap.add_argument("--batch-output-root", type=Path, default=None, help="TraceForge batch output root to render PNG overlays")
    ap.add_argument("--adapter-root", type=Path, default=None, help="Optional adapted episode root for semantic preview PNGs")
    ap.add_argument("--vis-output-dir", type=Path, default=None, help="Output dir for batch PNG visualizations")
    ap.add_argument("--max-samples-per-episode", type=int, default=2)
    ap.add_argument(
        "--field",
        type=str,
        default="traj_2d",
        help="2D trajectory field inside the sample npz, e.g. traj_2d or traj_2d_framewise_projected",
    )
    ap.add_argument("--episode-dir", type=Path, default=None, help="Episode output dir; defaults to sample parent parent")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--linewidth", type=int, default=3)
    ap.add_argument("--tracks-leave-trace", type=int, default=-1)
    ap.add_argument("--show-first-frame", type=int, default=10)
    ap.add_argument("--output", type=Path, default=None, help="Output mp4 path; defaults near the sample npz")
    args = ap.parse_args()

    if args.batch_output_root is not None:
        out_dir = args.vis_output_dir or (args.batch_output_root / "bridge_visualizations")
        outputs = []
        if args.adapter_root is not None:
            outputs += make_adapter_previews(args.adapter_root, out_dir / "adapter_previews")
        outputs += make_batch_overlays(args.batch_output_root, out_dir / "traceforge_overlays", args.max_samples_per_episode)
        outputs += make_flow_summaries(args.batch_output_root, out_dir / "flow_summaries")
        manifest_path = out_dir / "visualization_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"outputs": [str(p) for p in outputs]}, indent=2), encoding="utf-8")
        print(f"[Done] wrote {len(outputs)} visualization files under {out_dir}")
        return

    if args.sample_npz is None:
        raise SystemExit("Either --sample-npz or --batch-output-root is required")

    sample = load_sample(args.sample_npz)
    if args.field not in sample:
        raise ValueError(f"{args.sample_npz} does not contain field {args.field}")

    episode_dir = args.episode_dir if args.episode_dir is not None else args.sample_npz.parent.parent
    video_name = episode_dir.name
    images_dir = episode_dir / "images"

    traj_2d = np.asarray(sample[args.field]).astype(np.float32)
    if traj_2d.ndim != 3 or traj_2d.shape[-1] != 2:
        raise ValueError(f"Expected {args.field} shape (N,T,2), got {traj_2d.shape}")
    n_tracks, traj_len, _ = traj_2d.shape

    frame_index = int(np.asarray(sample["frame_index"]).reshape(-1)[0])
    valid_steps = sample.get("valid_steps")
    rgb_video = load_rgb_frames(images_dir, video_name, frame_index, traj_len)
    actual_len = int(rgb_video.shape[0])
    traj_2d = traj_2d[:, :actual_len, :]
    visibility = build_visibility(n_tracks, actual_len, valid_steps)
    traj_2d, visibility = sanitize_tracks(traj_2d.transpose(1, 0, 2), visibility)
    traj_2d = traj_2d.transpose(1, 0, 2)
    valids = np.ones((actual_len, n_tracks), dtype=bool)

    video_t = torch.from_numpy(rgb_video).permute(0, 3, 1, 2)[None].float()
    tracks_t = torch.from_numpy(traj_2d.transpose(1, 0, 2))[None].float()
    visibility_t = torch.from_numpy(visibility)[None]
    valids_t = torch.from_numpy(valids)[None]

    output_path = args.output
    if output_path is None:
        output_path = args.sample_npz.with_name(f"{args.sample_npz.stem}_{args.field}.mp4")
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    vis = Visualizer(
        save_dir=str(output_dir),
        fps=args.fps,
        linewidth=args.linewidth,
        mode="cool",
        show_first_frame=args.show_first_frame,
        tracks_leave_trace=args.tracks_leave_trace,
    )
    vis.visualize(
        video=video_t,
        tracks=tracks_t,
        visibility=visibility_t,
        valids=valids_t,
        filename=output_path.stem,
    )
    print(f"[Done] wrote 2D flow video to {output_path}")


if __name__ == "__main__":
    main()
