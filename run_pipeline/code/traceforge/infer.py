import os
import sys
import numpy as np
import cv2
import mediapy as media
import torch
from PIL import Image
import math
import tqdm
import glob
from rich import print
import argparse
from loguru import logger
import json

TRACEFORGE_ROOT = os.environ.get("TRACEFORGE_ROOT", "/home/zhy/data/TraceForge")
if TRACEFORGE_ROOT and TRACEFORGE_ROOT not in sys.path:
    sys.path.insert(0, TRACEFORGE_ROOT)

from utils.video_depth_pose_utils import video_depth_pose_dict
from utils.role_query_utils import RoleQuerySampler
from datasets.data_ops import _filter_one_depth
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from utils.inference_utils import load_model, inference
from utils.threed_utils import (
    project_tracks_3d_to_2d,
    project_tracks_3d_to_3d,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to video directory (for batch processing) or single video folder",
    )
    parser.add_argument(
        "--depth_path",
        type=str,
        default=None,
        help="Path to depth directory (if known depth is provided) for batch processing or single video folder",
    )
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument(
        "--input_layout",
        type=str,
        default="auto",
        choices=["auto", "generic_frames", "rvideo_traj_dataset"],
        help="Input layout. Use rvideo_traj_dataset for /datasets/<task>/traj_xxx with rgb_*.png/depth_*.png/camera_in.npy",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/tapip3d_final.pth"
    )
    parser.add_argument('--depth_pose_method', type=str, default='vggt4', choices=video_depth_pose_dict.keys())
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_iters", type=int, default=6)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--max_num_frames", type=int, default=384)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="Trajectory horizon length for each sample",
    )
    parser.add_argument(
        "--batch_process",
        action="store_true",
        default=False,
        help="Process all video folders in the given directory",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip processing if output already exists",
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        default="",
        help="Optional comma-separated task names when using rvideo_traj_dataset layout.",
    )
    parser.add_argument(
        "--use_all_trajectories",
        action="store_true",
        default=True,
        help="Include all visible trajectories in each frame (default: True)",
    )
    parser.add_argument(
        "--frame_drop_rate",
        type=int,
        default=1,
        help="Query uniform grid points every N frames (default: 1, query every frame)",
    )
    parser.add_argument(
        "--scan_depth",
        type=int,
        default=2,  # default depth changed to 2
        help="How many directory levels below --video_path to scan for subfolders "
            "when --batch_process is enabled. Default is 2 (e.g., P02_02_01)."
    )
    parser.add_argument(
        "--future_len",
        type=int,
        default=128,
        help="Tracking window length (number of frames) per query frame in offline mode",
    )
    parser.add_argument(
        "--auto_raise_future_len_to_seq_len",
        action="store_true",
        default=False,
        help="If future_len is smaller than model.seq_len, raise the effective tracking window to model.seq_len instead of failing.",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=50,
        help="Target max frames to keep per episode. If --fps <= 0, use stride = ceil(N / max_frames_per_video).",
    )
    parser.add_argument(
        "--query_mode",
        type=str,
        default="grid",
        choices=["grid", "masked_roles"],
        help="How to generate queries for TraceForge. grid keeps official behavior; masked_roles constrains queries to hand/tool/target masks.",
    )
    parser.add_argument(
        "--masked_role_fallback",
        type=str,
        default="fail",
        choices=["fail", "grid"],
        help="Behavior when masked role query generation fails. fail preserves strict masked-role behavior; grid falls back for that query frame.",
    )
    parser.add_argument(
        "--masked_query_points",
        type=int,
        default=400,
        help="Total masked role-constrained queries per query frame.",
    )
    parser.add_argument(
        "--query_mask_dilate_px",
        type=int,
        default=5,
        help="Dilation radius in pixels for role masks before query sampling.",
    )
    parser.add_argument(
        "--stationary_camera",
        action="store_true",
        default=False,
        help="Assume a fixed camera and repeat the first estimated pose across frames.",
    )
    parser.add_argument(
        "--use_known_intrinsics",
        action="store_true",
        default=False,
        help="Replace predicted intrinsics with camera_in.npy when available.",
    )
    parser.add_argument(
        "--use_known_depth",
        action="store_true",
        default=False,
        help="Use depth_*.png as known depth input when available. Default keeps TraceForge's internal depth estimation path.",
    )
    parser.add_argument(
        "--rgb_prefix",
        type=str,
        default="rgb",
        help="Frame prefix for RGB images in rvideo_traj_dataset mode.",
    )
    parser.add_argument(
        "--depth_prefix",
        type=str,
        default="depth",
        help="Frame prefix for depth images in rvideo_traj_dataset mode.",
    )
    parser.add_argument(
        "--rvideo_pipeline_root",
        type=str,
        default="/home/zhy/data/run_pipeline/code",
        help="Local code root for bundled detector helpers. Kept for compatibility.",
    )
    parser.add_argument(
        "--sam2_repo_root",
        type=str,
        default="/home/zhy/data/sam2",
    )
    parser.add_argument(
        "--groundingdino_repo_root",
        type=str,
        default="/home/zhy/data/GroundingDINO",
    )
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument(
        "--sam2_ckpt",
        type=str,
        default="/home/zhy/data/sam2/checkpoints/sam2.1_hiera_large.pt",
    )
    parser.add_argument(
        "--dino_cfg",
        type=str,
        default="/home/zhy/data/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--dino_ckpt",
        type=str,
        default="/home/zhy/data/GroundingDINO/checkpoint/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--dino_box_thresh", type=float, default=0.30)
    parser.add_argument("--dino_text_thresh", type=float, default=0.30)
    return parser.parse_args()

def retarget_trajectories(
    trajectory: np.ndarray,
    interval: float = 0.05,
    max_length: int = 64,
    top_percent: float = 0.02,
):
    """
    Synchronous arc-length retargeting using per-segment robust speeds.

    Steps:
      1) Global normalize x,y by (trajectory[-1,0,0], trajectory[-1,0,1]), then clip x,y to [0,1].
      2) For each time segment t: compute lengths for all tracks; take mean of top `top_percent`
         → robust_seglen[t].
      3) Build cumulative arc-length from robust_seglen and place targets every `interval`.
         (Long segments get subdivided; short ones merge implicitly.)
      4) For each target in segment t with fraction alpha, interpolate *all* tracks
         between frames t and t+1 with the same alpha (synchronous).
      5) Denormalize x,y only; z (if present) is linearly interpolated without scaling.

    Args:
        trajectory: (N, H, D) with D in {2,3}
        interval: target arc-length step
        max_length: output max length
        top_percent: fraction (0,1] for robust top-k mean per segment (e.g., 0.02 = top 2%)

    Returns:
        retargeted: (N, max_length, D), padded with -np.inf
        valid_mask: (max_length) bool
    """
    assert trajectory.ndim == 3, "trajectory must be (N, H, D)"
    N, H, D = trajectory.shape
    assert D in (2, 3), "D must be 2 or 3"
    if not (0 < top_percent <= 1.0):
        raise ValueError("top_percent must be in (0, 1].")
    if interval <= 0:
        raise ValueError("interval must be > 0")
    if H < 2:
        # If H==1, there is no segment to interpolate → return only the first frame
        ret = np.full((N, max_length, D), -np.inf, dtype=trajectory.dtype)
        mask = np.zeros((max_length), dtype=bool)
        ret[:, 0, :] = trajectory[:, 0, :]
        mask[0] = True
        return ret, mask

    eps = 1e-12

    # ---- 1) Global normalization (x,y) & clipping ----
    scale_x = float(trajectory[-1, 0, 0])
    scale_y = float(trajectory[-1, 0, 1])
    if abs(scale_x) < eps: scale_x = 1.0
    if abs(scale_y) < eps: scale_y = 1.0

    traj_norm = trajectory.astype(np.float64, copy=True)
    traj_norm[:, :, 0] /= scale_x
    traj_norm[:, :, 1] /= scale_y
    # clip x,y to [0,1]
    np.clip(traj_norm[:, :, 0], 0.0, 1.0, out=traj_norm[:, :, 0])
    np.clip(traj_norm[:, :, 1], 0.0, 1.0, out=traj_norm[:, :, 1])
    # z is not scaled/clipped

    # ---- 2) Robust length per segment t: mean of top k% ----
    # seglens_all: (N, H-1)
    diffs_all = traj_norm[:, 1:, :] - traj_norm[:, :-1, :]
    seglens_all = np.linalg.norm(diffs_all, axis=2)

    k = max(1, int(np.ceil(top_percent * N)))
    # Use np.partition to get per-segment (column-wise) top-k without full sorting
    # Values below index N-k are smaller; values at/above are larger
    part = np.partition(seglens_all, N - k, axis=0)      # (N, H-1)
    topk = part[N - k:, :]                                # (k, H-1)
    robust_seglen = topk.mean(axis=0)                     # (H-1,)

    total_len = float(robust_seglen.sum())
    # Output buffers
    retargeted = np.full((N, max_length, D), -np.inf, dtype=trajectory.dtype)
    valid_mask = np.zeros((max_length), dtype=bool)

    # ---- 3) Create targets at 'interval' along the robust cumulative length ----
    k_max = int(np.floor(total_len / interval))
    num_samples = min(k_max + 1, max_length)
    targets = interval * np.arange(num_samples, dtype=np.float64)
    targets[-1] = min(targets[-1], total_len)

    # Cumulative length s (vertex-based): s[0]=0, s[i]=sum_{j<i} robust_seglen[j]
    s = np.zeros((H,), dtype=np.float64)
    s[1:] = np.cumsum(robust_seglen, dtype=np.float64)

    # Segment index and in-segment fraction alpha for each target
    idx_seq = np.searchsorted(s, targets, side='right') - 1   # (num_samples,)
    idx_seq = np.clip(idx_seq, 0, H - 2)
    denom = np.maximum(robust_seglen[idx_seq], eps)           # (num_samples,)
    alpha = (targets - s[idx_seq]) / denom                    # (num_samples,)
    alpha_seq = alpha.reshape(-1, 1)                          # (num_samples,1)

    # ---- 4) Synchronous interpolation: apply the same (idx, alpha) to all tracks ----
    left = traj_norm[:, idx_seq, :]           # (N, num_samples, D)
    right = traj_norm[:, idx_seq + 1, :]      # (N, num_samples, D)
    samples_norm = left + alpha_seq[None, :, :] * (right - left)  # (N, num_samples, D)

    # ---- 5) Denormalize: scale only x,y back ----
    samples_out = samples_norm.astype(trajectory.dtype, copy=True)
    samples_out[:, :, 0] *= scale_x
    samples_out[:, :, 1] *= scale_y
    # Keep z as the linear interpolation result

    L = num_samples
    retargeted[:, :L, :] = samples_out
    valid_mask[:L] = True
    return retargeted, valid_mask


def save_structured_data(
    video_name,
    output_dir,
    video_tensor,
    depths,
    coords,
    visibs,
    intrinsics,
    extrinsics,
    query_points_per_frame,
    horizon,
    original_filenames,
    use_all_trajectories=True,
    query_frame_results=None,
    future_len: int = 128,
    role_id_to_name=None,
):
    """Save data in the structured format.

    Key design choice:
    - samples/*.npz remain the primary trajectory export
    - images/ and depth/ now store all per-segment frames using processed frame indices
      so sample-level visualizers can read continuous frames [start, start+1, ...].
    """

    video_output_dir = os.path.join(output_dir, video_name)
    images_dir = os.path.join(video_output_dir, "images")
    depth_dir = os.path.join(video_output_dir, "depth")
    samples_dir = os.path.join(video_output_dir, "samples")

    for dir_path in [images_dir, depth_dir, samples_dir]:
        os.makedirs(dir_path, exist_ok=True)

    role_id_to_name = role_id_to_name or {0: "generic"}

    if query_frame_results is not None:
        logger.info(f"Processing {len(query_frame_results)} query frame results")

        saved_count = 0
        all_retargeted = []
        all_role_ids = []
        all_query_frames = []
        role_queries_dir = os.path.join(video_output_dir, "role_queries")
        role_masks_dir = os.path.join(video_output_dir, "role_masks")
        os.makedirs(role_queries_dir, exist_ok=True)
        os.makedirs(role_masks_dir, exist_ok=True)

        for query_frame_idx, frame_data in query_frame_results.items():
            coords_np = frame_data["coords"].cpu().numpy()
            video_segment = frame_data["video_segment"].cpu().numpy() * 255.0
            video_segment = video_segment.astype(np.uint8).transpose(0, 2, 3, 1)
            depths_segment = frame_data["depths_segment"].cpu().numpy()

            visibs_np = frame_data["visibs"].cpu().numpy()
            intrinsics_np = frame_data["intrinsics_segment"].cpu().numpy()
            extrinsics_np = frame_data["extrinsics_segment"].cpu().numpy()

            logger.debug(f"Query frame {query_frame_idx}: coords_np shape = {coords_np.shape}")
            logger.debug(f"Query frame {query_frame_idx}: visibs_np shape = {visibs_np.shape}")

            if len(coords_np.shape) != 3:
                logger.error(f"Unexpected coords shape for frame {query_frame_idx}: {coords_np.shape}")
                continue

            actual_frames = int(coords_np.shape[0])
            sample_data = {}

            frame_h, frame_w = video_segment.shape[1:3]
            keypoints = np.asarray(
                frame_data.get(
                    "query_points_xy",
                    query_points_per_frame.get(query_frame_idx, np.zeros((0, 2), dtype=np.float32)),
                ),
                dtype=np.float32,
            )
            query_role_ids = frame_data.get("query_role_ids")
            if query_role_ids is None:
                query_role_ids = np.zeros((keypoints.shape[0],), dtype=np.int32)
            query_role_ids = np.asarray(query_role_ids, dtype=np.int32)

            sample_data["image_path"] = np.array(
                [f"images/{video_name}_{query_frame_idx}.png"], dtype="<U80"
            )
            sample_data["frame_index"] = np.array([query_frame_idx], dtype=np.int32)
            sample_data["keypoints"] = keypoints.astype(np.float32)

            try:
                sample_data["traj"] = coords_np.transpose(1, 0, 2).astype(np.float32)
            except ValueError as e:
                logger.error(f"Error transposing coords for frame {query_frame_idx}: {e}")
                logger.error(f"coords_np shape: {coords_np.shape}")
                continue

            camera_views_segment = []
            for t in range(len(intrinsics_np)):
                camera_views_segment.append(
                    {
                        "c2w": np.linalg.inv(extrinsics_np[t]),
                        "K": intrinsics_np[t],
                        "height": frame_h,
                        "width": frame_w,
                    }
                )

            fixed_camera_view = camera_views_segment[0]
            coords_3d_for_projection = coords_np
            try:
                tracks2d_fixed = project_tracks_3d_to_2d(
                    tracks3d=coords_3d_for_projection,
                    camera_views=[fixed_camera_view] * len(coords_3d_for_projection),
                )
                tracks3d_fixed = project_tracks_3d_to_3d(
                    tracks3d=coords_3d_for_projection,
                    camera_views=[fixed_camera_view] * len(coords_3d_for_projection),
                )
                sample_data["traj_2d"] = tracks2d_fixed.transpose(1, 0, 2).astype(np.float32)
                sample_data["traj"] = tracks3d_fixed.transpose(1, 0, 2).astype(np.float32)
            except Exception as e:
                logger.error(f"Error projecting tracks for frame {query_frame_idx}: {e}")
                sample_data["traj_2d"] = coords_np[:, :, :2].transpose(1, 0, 2).astype(np.float32)
                sample_data["traj"] = coords_np.transpose(1, 0, 2).astype(np.float32)

            # Save all processed segment frames using contiguous processed-frame indices.
            # This keeps images/ compatible with sample visualizers that expect
            # frame_index, frame_index+1, ... to exist.
            for local_t in range(actual_frames):
                processed_frame_idx = query_frame_idx + local_t

                img_filename = f"{video_name}_{processed_frame_idx}.png"
                img_path = os.path.join(images_dir, img_filename)
                if not os.path.exists(img_path):
                    Image.fromarray(video_segment[local_t]).save(img_path)

                depth_filename = f"{video_name}_{processed_frame_idx}.png"
                depth_path = os.path.join(depth_dir, depth_filename)
                if not os.path.exists(depth_path):
                    depth_frame = depths_segment[local_t]
                    depth_normalized = (depth_frame * 10000).astype(np.uint16)
                    Image.fromarray(depth_normalized, mode="I;16").save(depth_path)

                depth_raw_filename = f"{video_name}_{processed_frame_idx}_raw.npz"
                depth_raw_path = os.path.join(depth_dir, depth_raw_filename)
                if not os.path.exists(depth_raw_path):
                    np.savez(depth_raw_path, depth=depths_segment[local_t])

            retargeted, valid_mask = retarget_trajectories(
                sample_data["traj"], max_length=future_len
            )
            sample_data["traj"] = retargeted
            sample_data["valid_steps"] = valid_mask

            all_retargeted.append(retargeted.astype(np.float32))
            all_role_ids.append(query_role_ids.astype(np.int32))
            all_query_frames.append(
                np.full((retargeted.shape[0],), query_frame_idx, dtype=np.int32)
            )
            np.save(
                os.path.join(role_queries_dir, f"{video_name}_{query_frame_idx}_role_id.npy"),
                query_role_ids.astype(np.int32),
            )
            query_source_payload = {
                "query_frame_idx": int(query_frame_idx),
                "source": str(frame_data.get("query_source", "unknown")),
                "fallback_reason": str(frame_data.get("fallback_reason", "")),
            }
            with open(
                os.path.join(role_queries_dir, f"{video_name}_{query_frame_idx}_query_source.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(query_source_payload, f, indent=2)
            debug_masks = frame_data.get("debug_masks") or {}
            for role_name, mask in debug_masks.items():
                mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
                Image.fromarray(mask_u8).save(
                    os.path.join(role_masks_dir, f"{video_name}_{query_frame_idx}_{role_name}.png")
                )
            if keypoints.shape[0] > 0 and video_segment.shape[0] > 0:
                overlay = Image.fromarray(video_segment[0]).convert("RGB")
                try:
                    from PIL import ImageDraw

                    draw = ImageDraw.Draw(overlay)
                    colors = {
                        1: (255, 64, 64),
                        2: (64, 200, 255),
                        3: (80, 255, 120),
                        4: (255, 220, 64),
                    }
                    for point_xy, role_id in zip(keypoints[:: max(1, keypoints.shape[0] // 300)], query_role_ids[:: max(1, keypoints.shape[0] // 300)]):
                        x, y = float(point_xy[0]), float(point_xy[1])
                        c = colors.get(int(role_id), (255, 255, 255))
                        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=c)
                    overlay.save(os.path.join(role_queries_dir, f"{video_name}_{query_frame_idx}_queries.png"))
                except Exception as exc:
                    logger.warning(f"Failed to save query overlay for {video_name} frame {query_frame_idx}: {exc}")

            sample_filename = f"{video_name}_{query_frame_idx}.npz"
            sample_path = os.path.join(samples_dir, sample_filename)
            np.savez(sample_path, **sample_data)

            logger.info(
                f"Saved query frame {query_frame_idx} with {retargeted.shape[0]} trajectories tracked for {actual_frames} frames"
            )
            saved_count += 1

        with open(os.path.join(video_output_dir, "query_role_name.json"), "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in role_id_to_name.items()}, f, indent=2)

        if all_retargeted:
            selected_traces = np.concatenate(all_retargeted, axis=0).astype(np.float32)
            selected_role_ids = np.concatenate(all_role_ids, axis=0).astype(np.int32)
            selected_query_frames = np.concatenate(all_query_frames, axis=0).astype(np.int32)
            np.save(os.path.join(video_output_dir, "selected_traces.npy"), selected_traces)
            np.save(os.path.join(video_output_dir, "query_role_id.npy"), selected_role_ids)
            np.save(os.path.join(video_output_dir, "query_frame_index.npy"), selected_query_frames)
            for role_id, role_name in role_id_to_name.items():
                if role_name == "generic":
                    continue
                role_mask = selected_role_ids == int(role_id)
                np.save(
                    os.path.join(video_output_dir, f"{role_name}_flow.npy"),
                    selected_traces[role_mask].astype(np.float32),
                )

        logger.info(f"Saved {saved_count} frames")

def detect_input_layout(video_path: str, args) -> str:
    if args.input_layout != "auto":
        return args.input_layout
    if os.path.isdir(video_path):
        rgb_matches = glob.glob(os.path.join(video_path, f"{args.rgb_prefix}_*.png"))
        if rgb_matches and os.path.exists(os.path.join(video_path, "camera_in.npy")):
            return "rvideo_traj_dataset"
        nested_rgb_matches = glob.glob(os.path.join(video_path, "*", "*", f"{args.rgb_prefix}_*.png"))
        if nested_rgb_matches:
            return "rvideo_traj_dataset"
    return "generic_frames"


def load_rvideo_traj_episode(
    traj_dir,
    rgb_prefix="rgb",
    depth_prefix="depth",
    fps=1,
    max_num_frames=384,
    load_depth=False,
    load_intrinsics=False,
):
    rgb_files = sorted(
        glob.glob(os.path.join(traj_dir, f"{rgb_prefix}_*.png")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]),
    )
    if not rgb_files:
        raise FileNotFoundError(f"No RGB frames with prefix '{rgb_prefix}_' found under {traj_dir}")
    rgb_files = rgb_files[::fps][:max_num_frames]

    video_tensor = []
    original_filenames = []
    frame_indices = []
    for rgb_path in rgb_files:
        image = Image.open(rgb_path).convert("RGB")
        video_tensor.append(torch.from_numpy(np.asarray(image)).float())
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        original_filenames.append(stem)
        frame_indices.append(int(stem.split("_")[-1]))

    video_tensor = torch.stack(video_tensor).permute(0, 3, 1, 2).float() / 255.0

    depth_tensor = None
    if load_depth:
        depth_available = True
        depth_list = []
        for frame_idx in frame_indices:
            depth_path = os.path.join(traj_dir, f"{depth_prefix}_{frame_idx}.png")
            if not os.path.exists(depth_path):
                depth_available = False
                break
            depth_img = Image.open(depth_path).convert("I;16")
            depth_list.append(torch.from_numpy(np.asarray(depth_img)).float())
        if depth_available and depth_list:
            depth_tensor = torch.stack(depth_list)
            valid_depth = depth_tensor > 0
            depth_tensor[~valid_depth] = 0

    intrinsics = None
    if load_intrinsics:
        intr_path = os.path.join(traj_dir, "camera_in.npy")
        if os.path.exists(intr_path):
            intrinsics = np.asarray(np.load(intr_path), dtype=np.float32)

    return {
        "video_tensor": video_tensor,
        "depth_tensor": depth_tensor,
        "known_intrinsics": intrinsics,
        "original_filenames": original_filenames,
        "frame_indices": frame_indices,
    }


def process_single_video(video_path, depth_path, args, model_3dtracker, model_depth_pose):
    """Process a single video and return the processed data"""
    logger.info(f"Processing video: {video_path}")
    input_layout = detect_input_layout(video_path, args)
    role_query_sampler = RoleQuerySampler(args) if args.query_mode == "masked_roles" else None

    # --- NEW: per-episode stride based on frame count when --fps <= 0 ---
    # If user set --fps > 0, use that fixed stride; otherwise auto-compute from N.
    if args.fps and int(args.fps) > 0:
        stride = int(args.fps)
        n_frames = 0  # unknown/not needed in fixed stride mode
    else:
        stride = 1
        n_frames = 0
        if os.path.isdir(video_path):
            # Count frames by scanning image files in the episode folder
            img_files = []
            if input_layout == "rvideo_traj_dataset":
                img_files.extend(glob.glob(os.path.join(video_path, f"{args.rgb_prefix}_*.png")))
            else:
                for ext in ["jpg", "jpeg", "png"]:
                    img_files.extend(glob.glob(os.path.join(video_path, f"*.{ext}")))
            n_frames = len(img_files)

            # Auto stride: ceil(N / target), where target = --max_frames_per_video
            target = max(1, int(getattr(args, "max_frames_per_video", 150)))
            stride = max(1, math.ceil(n_frames / target)) if n_frames > 0 else 1
        else:
            # For video files (.mp4, etc.), we keep stride=1 (or you can extend to probe length)
            stride = 1

    logger.info(
        f"[{os.path.basename(video_path)}] frames={n_frames if n_frames else 'n/a'} "
        f"target={getattr(args, 'max_frames_per_video', 150)} -> stride={stride}"
    )

    known_intrinsics = None
    frame_indices = list(range(n_frames)) if n_frames > 0 else []
    if input_layout == "rvideo_traj_dataset":
        episode = load_rvideo_traj_episode(
            video_path,
            rgb_prefix=args.rgb_prefix,
            depth_prefix=args.depth_prefix,
            fps=stride,
            max_num_frames=args.max_num_frames,
            load_depth=bool(args.use_known_depth),
            load_intrinsics=bool(args.use_known_intrinsics),
        )
        video_tensor = episode["video_tensor"]
        depth_tensor = episode["depth_tensor"]
        known_intrinsics = episode["known_intrinsics"]
        original_filenames = episode["original_filenames"]
        frame_indices = episode["frame_indices"]
        video_mask = None
    else:
        # Load RGB with computed stride
        video_tensor, video_mask, original_filenames = load_video_and_mask(
            video_path, args.mask_dir, stride, args.max_num_frames
        )

        # Load depth (if provided) with the SAME stride to keep alignment with RGB
        depth_tensor = None
        if depth_path is not None:
            depth_tensor, _, _ = load_video_and_mask(
                depth_path, None, stride, args.max_num_frames, is_depth=True
            )  # [T, H, W]
            valid_depth = depth_tensor > 0
            depth_tensor[~valid_depth] = 0  # Invalidate bad depth values
        if not frame_indices:
            frame_indices = list(range(len(video_tensor)))

    video_length = len(video_tensor)

    # obtain video depth and pose
    (
        video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy
    ) = model_depth_pose(
        video_tensor,
        known_depth=depth_tensor if bool(args.use_known_depth) else None,
        known_intrinsics=known_intrinsics,
        stationary_camera=bool(args.stationary_camera),
        replace_with_known_depth=False,  # if known depth is given, always replace
        replace_with_known_intrinsics=bool(args.use_known_intrinsics),
    )

    # Keep depth_conf for visualization NPZ
    depth_conf_npy = np.asarray(depth_conf)

    frame_H, frame_W = video_ten.shape[-2:]
    original_frame_h, original_frame_w = video_tensor.shape[-2:]
    model_seq_len = int(getattr(model_3dtracker, "seq_len", 0) or 0)
    effective_future_len = int(args.future_len)
    if model_seq_len > 0 and effective_future_len < model_seq_len:
        if bool(getattr(args, "auto_raise_future_len_to_seq_len", False)):
            logger.warning(
                f"future_len={effective_future_len} is smaller than model.seq_len={model_seq_len}; "
                f"raising effective future_len to {model_seq_len} so the tracker enters at least one tracking window."
            )
            effective_future_len = model_seq_len
        else:
            raise ValueError(
                f"future_len={effective_future_len} is smaller than model.seq_len={model_seq_len}. "
                "This would prevent PointTracker3D from entering any tracking window and would yield static coords. "
                "Set --future_len >= model.seq_len or pass --auto_raise_future_len_to_seq_len."
            )

    query_points_per_frame = {}
    query_point = []
    query_role_ids_per_frame = {}
    role_debug_masks_per_frame = {}
    query_source_per_frame = {}
    fallback_reason_per_frame = {}
    role_id_to_name_global = {0: "generic"}
    role_prompts = None
    tracking_segments = []  # Store info about which frames to track for each segment

    # Determine which frames to query based on frame_drop_rate
    query_frames = list(range(0, video_length, args.frame_drop_rate))
    logger.info(f"Using query_mode={args.query_mode} on frames: {query_frames} (frame_drop_rate={args.frame_drop_rate})")
    logger.info(f"Tracking up to {effective_future_len} frames from each query frame")

    for frame_idx in query_frames:
        # Calculate the end frame for this tracking segment (16 frames max)
        end_frame = min(frame_idx + effective_future_len, video_length)
        tracking_segments.append((frame_idx, end_frame))

        if args.query_mode == "masked_roles":
            rgb_frame = (video_tensor[frame_idx].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            if input_layout == "rvideo_traj_dataset" and frame_idx < len(frame_indices):
                image_path = os.path.join(video_path, f"{args.rgb_prefix}_{frame_indices[frame_idx]}.png")
            else:
                image_path = video_path if os.path.isfile(video_path) else os.path.join(video_path, original_filenames[frame_idx] + ".png")
            try:
                role_result = role_query_sampler.sample_queries(
                    frame_idx=frame_idx,
                    image_rgb=rgb_frame,
                    image_path=image_path,
                    traj_dir=video_path,
                    frame_h=frame_H,
                    frame_w=frame_W,
                )
                query_point.append(role_result.query_points)
                query_points_per_frame[frame_idx] = role_result.query_points[:, 1:3].astype(np.float32)
                query_role_ids_per_frame[frame_idx] = role_result.query_role_ids.astype(np.int32)
                role_debug_masks_per_frame[frame_idx] = role_result.debug_masks
                role_id_to_name_global.update(role_result.role_id_to_name)
                role_prompts = role_result.prompts
                query_source_per_frame[frame_idx] = "masked_roles"
            except Exception as exc:
                if str(getattr(args, "masked_role_fallback", "fail")) != "grid":
                    raise
                logger.warning(
                    f"masked_roles failed on frame {frame_idx}; falling back to grid queries for this frame: {exc}"
                )
                grid_points = (
                    create_uniform_grid_points(
                        height=frame_H, width=frame_W, grid_size=20, device="cpu"
                    )
                    .squeeze(0)
                    .numpy()
                )
                grid_points[:, 0] = frame_idx
                query_point.append(grid_points)
                query_points_per_frame[frame_idx] = grid_points[:, 1:3].astype(np.float32)
                query_role_ids_per_frame[frame_idx] = np.zeros((grid_points.shape[0],), dtype=np.int32)
                role_debug_masks_per_frame[frame_idx] = {}
                query_source_per_frame[frame_idx] = "fallback_grid"
                fallback_reason_per_frame[frame_idx] = str(exc)
        else:
            grid_points = (
                create_uniform_grid_points(
                    height=frame_H, width=frame_W, grid_size=20, device="cpu"
                )
                .squeeze(0)
                .numpy()
            )
            grid_points[:, 0] = frame_idx
            query_point.append(grid_points)
            query_points_per_frame[frame_idx] = grid_points[:, 1:3].astype(np.float32)

    if args.query_mode != "masked_roles":
        role_id_to_name_global = {0: "generic"}

    # Process each query frame independently with 16-frame tracking
    extrs_npy = np.linalg.inv(extrs_npy)

    # Store results for each query frame
    query_frame_results = {}

    logger.info(f"Processing {len(tracking_segments)} independent tracking segments")

    for seg_idx, (start_frame, end_frame) in enumerate(tracking_segments):
        logger.info(
            f"Processing query frame {start_frame}: tracking {end_frame - start_frame} frames"
        )

        # Clear CUDA cache before each segment to avoid fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract video segment (16 frames starting from query frame)
        video_segment = video_ten[start_frame:end_frame]
        depth_segment = depth_npy[start_frame:end_frame]
        intrs_segment = intrs_npy[start_frame:end_frame]
        extrs_segment = extrs_npy[start_frame:end_frame]

        # Get query points for this segment (only from the starting frame)
        # Need to adjust the frame index to be relative to segment start (0)
        segment_query_point = [query_point[seg_idx].copy()]
        segment_query_point[0][:, 0] = 0  # Set frame index to 0 for segment start

        video, depths, intrinsics, extrinsics, query_point_tensor, support_grid_size, remapped_query_point = (
            prepare_inputs(
                video_segment,
                depth_segment,
                intrs_segment,
                extrs_segment,
                segment_query_point,
                inference_res=(frame_H, frame_W),
                support_grid_size=16,
                original_hw=(original_frame_h, original_frame_w),
                device=args.device,
            )
        )

        model_3dtracker.set_image_size((frame_H, frame_W))

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                coords_seg, visibs_seg = inference(
                    model=model_3dtracker,
                    video=video,
                    depths=depths,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    query_point=query_point_tensor,
                    num_iters=args.num_iters,
                    grid_size=support_grid_size,
                    bidrectional=False,  # Disable backward tracking
                )

        # Validate inference results before storing
        logger.debug(
            f"Query frame {start_frame}: coords_seg shape = {coords_seg.shape}, visibs_seg shape = {visibs_seg.shape}"
        )

        # Check if results have expected dimensions
        if len(coords_seg.shape) != 3 or len(visibs_seg.shape) != 2:
            logger.error(
                f"Query frame {start_frame}: Invalid result shapes - coords: {coords_seg.shape}, visibs: {visibs_seg.shape}"
            )
            continue

        # Store results for this query frame
        query_frame_results[start_frame] = {
            "coords": coords_seg,
            "visibs": visibs_seg,
            "video_segment": video,
            "depths_segment": depths,
            "intrinsics_segment": intrinsics,
            "extrinsics_segment": extrinsics,
            "query_points_xy": remapped_query_point[0][:, 1:3].astype(np.float32),
            "query_role_ids": query_role_ids_per_frame.get(start_frame),
            "debug_masks": role_debug_masks_per_frame.get(start_frame, {}),
            "query_source": query_source_per_frame.get(start_frame, "grid" if args.query_mode != "masked_roles" else "unknown"),
            "fallback_reason": fallback_reason_per_frame.get(start_frame, ""),
        }

        logger.info(
            f"Query frame {start_frame}: tracked {coords_seg.shape[1]} trajectories for {coords_seg.shape[0]} frames"
        )
        
        # Clear cache after inference to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # For compatibility with the rest of the pipeline, use the first segment as the main result
    # But we'll save each segment independently in save_structured_data
    if query_frame_results:
        first_frame = min(query_frame_results.keys())
        coords = query_frame_results[first_frame]["coords"]
        visibs = query_frame_results[first_frame]["visibs"]
        video = query_frame_results[first_frame]["video_segment"]
        depths = query_frame_results[first_frame]["depths_segment"]
        intrinsics = query_frame_results[first_frame]["intrinsics_segment"]
        extrinsics = query_frame_results[first_frame]["extrinsics_segment"]
    else:
        flen = min(effective_future_len, len(video_ten))
        coords = torch.empty((0, 0, 3))
        visibs = torch.empty((0, 0))
        video = video_ten[:flen]
        depths = torch.from_numpy(depth_npy[:flen]).float().to(args.device)
        intrinsics = torch.from_numpy(intrs_npy[:flen]).float().to(args.device)
        extrinsics = torch.from_numpy(extrs_npy[:flen]).float().to(args.device)

    # Validate tensor shapes after inference
    logger.debug(
        f"After inference - coords shape: {coords.shape}, visibs shape: {visibs.shape}"
    )

    # Ensure visibs has the expected dimensions
    if visibs.dim() == 3 and visibs.shape[-1] == 1:
        visibs = visibs.squeeze(-1)  # Remove last dimension if it's 1
        logger.debug(f"Squeezed visibs shape: {visibs.shape}")

    # Validate final shapes
    expected_frames = video.shape[0]
    expected_points = coords.shape[1] if coords.dim() >= 2 else 0
    if coords.dim() != 3 or visibs.dim() != 2:
        logger.error(
            f"Unexpected tensor dimensions - coords: {coords.shape}, visibs: {visibs.shape}"
        )
        raise ValueError(f"Invalid tensor shapes after inference")

    if coords.shape[0] != expected_frames or visibs.shape[0] != expected_frames:
        logger.error(
            f"Frame count mismatch - expected {expected_frames}, got coords: {coords.shape[0]}, visibs: {visibs.shape[0]}"
        )
        raise ValueError(f"Frame count mismatch in inference results")

    return {
        "video_tensor": video,
        "depths": depths,
        "coords": coords,
        "visibs": visibs,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "query_points_per_frame": query_points_per_frame,
        "original_filenames": original_filenames,
        "depth_conf": depth_conf_npy,
        "query_frame_results": query_frame_results,  # Add individual frame results
        "query_role_ids_per_frame": query_role_ids_per_frame,
        "role_id_to_name": role_id_to_name_global,
        "role_prompts": role_prompts,
        "full_intrinsics": torch.from_numpy(intrs_npy)
        .float()
        .to(args.device),  # Full video intrinsics
        "full_extrinsics": torch.from_numpy(extrs_npy)
        .float()
        .to(args.device),  # Full video extrinsics
        "effective_future_len": effective_future_len,
    }


def find_video_folders(base_path: str, scan_depth: int = 2):
    """
    Recursively scan subfolders up to a given depth and return inputs
    that contain images (.jpg/.jpeg/.png) or stand-alone video files
    (.mp4/.webm/etc.).

    Args:
        base_path: Root directory to scan
        scan_depth: Number of directory levels to traverse

    Returns:
        List of folder paths containing image files at the target depth
    """
    img_exts = (".jpg", ".jpeg", ".png")
    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg")

    # Normalize the base path
    base_path = os.path.abspath(base_path.rstrip(os.sep))
    base_depth = base_path.count(os.sep)
    target_depth = base_depth + scan_depth

    video_folders = []

    for root, dirs, files in os.walk(base_path):
        current_depth = os.path.abspath(root.rstrip(os.sep)).count(os.sep)

        # Skip folders above the target depth
        if current_depth < target_depth:
            continue

        # Select only folders/files exactly at the target depth
        if current_depth == target_depth:
            has_images = any(f.lower().endswith(img_exts) for f in files)
            if has_images:
                video_folders.append(root)
            # Also collect individual video files at this depth
            for f in files:
                if f.lower().endswith(video_exts):
                    video_folders.append(os.path.join(root, f))

        # Skip deeper folders for performance (no need to go further)
        if current_depth > target_depth:
            dirs[:] = []  # prevent os.walk from descending further

    # Deduplicate and sort for stable ordering
    video_folders = sorted(list(dict.fromkeys(video_folders)))
    return video_folders


def find_rvideo_traj_dirs(base_path: str, task_filter: str = ""):
    requested = {x.strip() for x in str(task_filter).split(",") if x.strip()}
    traj_dirs = []
    for task_name in sorted(os.listdir(base_path)):
        task_dir = os.path.join(base_path, task_name)
        if not os.path.isdir(task_dir):
            continue
        if requested and task_name not in requested:
            continue
        for traj_name in sorted(os.listdir(task_dir)):
            traj_dir = os.path.join(task_dir, traj_name)
            if not os.path.isdir(traj_dir):
                continue
            if glob.glob(os.path.join(traj_dir, "rgb_*.png")):
                traj_dirs.append(traj_dir)
    return traj_dirs


def load_video_and_mask(video_path, mask_dir=None, fps=1, max_num_frames=384, is_depth=False):
    original_filenames = []

    if os.path.isdir(video_path):
        img_files = []
        for ext in ["jpg", "png"]:
            img_files.extend(sorted(glob.glob(os.path.join(video_path, f"*.{ext}"))))

        # IMPORTANT: Subsample the file list BEFORE loading to save memory
        img_files = img_files[::fps]

        video_tensor = []
        for img_file in tqdm.tqdm(img_files, desc="Loading images"):
            img = Image.open(img_file)
            if is_depth:
                img = img.convert("I;16")  # 16-bit grayscale for depth
            else:
                img = img.convert("RGB")
            video_tensor.append(
                torch.from_numpy(np.array(img)).float()
            )
            # Extract original filename without extension
            filename = os.path.splitext(os.path.basename(img_file))[0]
            original_filenames.append(filename)
        video_tensor = torch.stack(video_tensor)  # (N, H, W, 3)
    elif video_path.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        # simple video reading. Please modify it if it causes OOM
        video_tensor = torch.from_numpy(media.read_video(video_path))
        # Generate frame names for video files
        for i in range(len(video_tensor)):
            original_filenames.append(f"frame_{i:010d}")
        # For video files, subsample after loading
        video_tensor = video_tensor[::fps]
        original_filenames = original_filenames[::fps]

    if not is_depth:
        video_tensor = video_tensor.permute(
            0, 3, 1, 2
        )  # Convert to tensor and permute to (N, C, H, W)
    video_tensor = video_tensor.float()
    video_tensor = video_tensor[:max_num_frames]
    original_filenames = original_filenames[:max_num_frames]
    video_length = len(video_tensor)
    logger.debug(f"Loaded video with {video_length} frames from {video_path}")
    frame_h, frame_w = video_tensor.shape[-2:]

    video_mask_npy = None
    if mask_dir is not None:
        video_mask_npy = []
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        for mask_file in mask_files:
            mask = media.read_image(mask_file)
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
            video_mask_npy.append(mask)
        video_mask_npy = np.stack(video_mask_npy)

    if not is_depth:
        video_tensor /= 255.
    return video_tensor, video_mask_npy, original_filenames


def create_uniform_grid_points(height, width, grid_size=20, device="cuda"):
    """Create uniform grid points across the image.

    Args:
        height (int): Image height
        width (int): Image width
        grid_size (int): Grid size (20x20)
        device (str): Device for tensor

    Returns:
        torch.Tensor: Grid points [1, grid_size*grid_size, 3] where each point is [t, x, y]
    """
    # Create uniform grid
    y_coords = np.linspace(0, height - 1, grid_size)
    x_coords = np.linspace(0, width - 1, grid_size)

    # Create meshgrid
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Flatten and create points [N, 2]
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)

    # Add time dimension (t=0 for all points) -> [N, 3]
    time_col = np.zeros((grid_points.shape[0], 1))
    grid_points_3d = np.concatenate([time_col, grid_points], axis=1)

    # Convert to tensor and add batch dimension -> [1, N, 3]
    grid_tensor = torch.tensor(
        grid_points_3d, dtype=torch.float32, device=device
    ).unsqueeze(0)

    return grid_tensor


def remap_query_points_to_processed_resolution(query_xyt, original_hw, processed_hw, target_size=518, keep_ratio=False):
    orig_h, orig_w = int(original_hw[0]), int(original_hw[1])
    proc_h, proc_w = int(processed_hw[0]), int(processed_hw[1])
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid original resolution: {original_hw}")

    new_w = target_size
    new_h = round(orig_h * (new_w / orig_w) / 14) * 14
    crop_offset_y = 0
    if not keep_ratio and new_h > target_size:
        crop_offset_y = (new_h - target_size) // 2

    remapped = []
    for query_i in query_xyt:
        if len(query_i) == 0:
            remapped.append(query_i)
            continue
        q = np.asarray(query_i, dtype=np.float32).copy()
        q[:, 1] = q[:, 1] * (float(new_w) / float(orig_w))
        q[:, 2] = q[:, 2] * (float(new_h) / float(orig_h)) - float(crop_offset_y)
        q[:, 1] = np.clip(q[:, 1], 0.0, max(0.0, float(proc_w - 1)))
        q[:, 2] = np.clip(q[:, 2], 0.0, max(0.0, float(proc_h - 1)))
        remapped.append(q)
    return remapped

def prepare_query_points(query_xyt, depths, intrinsics, extrinsics):
    final_queries = []
    for query_i in query_xyt:
        if len(query_i) == 0:
            continue

        t = int(query_i[0, 0])
        depth_t = depths[t]
        K_inv_t = np.linalg.inv(intrinsics[t])
        c2w_t = np.linalg.inv(extrinsics[t])

        xy = query_i[:, 1:]
        ji = np.round(xy).astype(int)
        ji[..., 0] = np.clip(ji[..., 0], 0, depth_t.shape[1] - 1)
        ji[..., 1] = np.clip(ji[..., 1], 0, depth_t.shape[0] - 1)
        d = depth_t[ji[..., 1], ji[..., 0]]
        xy_homo = np.concatenate([xy, np.ones_like(xy[:, :1])], axis=-1)
        local_coords = K_inv_t @ xy_homo.T  # (3, N)
        local_coords = local_coords * d[None, :]  # (3, N)
        world_coords = c2w_t[:3, :3] @ local_coords + c2w_t[:3, 3:]
        final_queries.append(np.concatenate([query_i[:, :1], world_coords.T], axis=-1))
    return np.concatenate(final_queries, axis=0)  # (N, 4)


def prepare_inputs(
    video_ten,
    depths,
    intrinsics,
    extrinsics,
    query_point,
    inference_res: Tuple[int, int],
    support_grid_size: int,
    original_hw: Tuple[int, int] | None = None,
    num_threads: int = 8,
    device: str = "cuda",
):
    _original_res = depths.shape[1:3]
    inference_res = _original_res  # fix as the same

    intrinsics[:, 0, :] *= (inference_res[1] - 1) / (_original_res[1] - 1)
    intrinsics[:, 1, :] *= (inference_res[0] - 1) / (_original_res[0] - 1)

    # resize & remove edges
    with ThreadPoolExecutor(num_threads) as executor:
        depths_futures = [
            executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic)
            for depth, intrinsic in zip(depths, intrinsics)
        ]
        depths = np.stack([future.result() for future in depths_futures])

    query_point = remap_query_points_to_processed_resolution(
        query_point,
        original_hw=original_hw if original_hw is not None else video_ten.shape[-2:],
        processed_hw=depths.shape[1:3],
    )
    remapped_query_point = [np.asarray(q, dtype=np.float32).copy() for q in query_point]
    query_point = prepare_query_points(remapped_query_point, depths, intrinsics, extrinsics)
    query_point = torch.from_numpy(query_point).float().to(device)
    video = (video_ten.float()).to(device).clamp(0, 1)
    depths = torch.from_numpy(depths).float().to(device)
    intrinsics = torch.from_numpy(intrinsics).float().to(device)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)

    return video, depths, intrinsics, extrinsics, query_point, support_grid_size, remapped_query_point


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # initialize 3D models
    model_depth_pose = video_depth_pose_dict[args.depth_pose_method](args)
    model_3dtracker = load_model(args.checkpoint).to(args.device)

    # Determine video paths to process
    if args.batch_process:
        batch_layout = args.input_layout
        if batch_layout == "auto":
            batch_layout = detect_input_layout(args.video_path, args)
        if batch_layout == "rvideo_traj_dataset":
            video_folders = find_rvideo_traj_dirs(args.video_path, args.task_filter)
            depth_folders = [None] * len(video_folders)
        else:
            video_folders = find_video_folders(args.video_path, args.scan_depth)
            if args.depth_path is not None:
                depth_folders = find_video_folders(args.depth_path)
                if len(depth_folders) != len(video_folders):
                    logger.error(
                        f"Number of depth folders ({len(depth_folders)}) does not match number of video folders ({len(video_folders)})"
                    )
                    exit(1)
            else:
                depth_folders = [None] * len(video_folders)

        logger.info(f"Found {len(video_folders)} video folders to process")
        if not video_folders:
            logger.error(f"No video folders found in {args.video_path}")
            exit(1)
    else:
        video_folders = [args.video_path]
        depth_folders = [args.depth_path]

    # Process each video
    for video_path, depth_path in zip(video_folders, depth_folders):
        if detect_input_layout(video_path, args) == "rvideo_traj_dataset":
            traj_name = os.path.basename(video_path.rstrip("/"))
            task_name = os.path.basename(os.path.dirname(video_path.rstrip("/")))
            video_name = f"{task_name}__{traj_name}"
        else:
            video_name = os.path.basename(video_path.rstrip("/"))

        # Check if output already exists and skip if requested
        if args.skip_existing:
            output_path = os.path.join(out_dir, video_name)
            if os.path.exists(output_path):
                logger.info(f"Skipping {video_name} - output already exists")
                continue

        try:
            # Clear CUDA cache before processing each video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process video
            result = process_single_video(video_path, depth_path, args, model_3dtracker, model_depth_pose)

            # Save structured data
            save_structured_data(
                video_name=video_name,
                output_dir=out_dir,
                video_tensor=result["video_tensor"],
                depths=result["depths"],
                coords=result["coords"],
                visibs=result["visibs"],
                intrinsics=result["intrinsics"],
                extrinsics=result["extrinsics"],
                query_points_per_frame=result["query_points_per_frame"],
                horizon=args.horizon,
                original_filenames=result["original_filenames"],
                use_all_trajectories=args.use_all_trajectories,
                query_frame_results=result.get("query_frame_results"),
                future_len=result.get("effective_future_len", args.future_len),
                role_id_to_name=result.get("role_id_to_name"),
            )

            # Always save traditional visualization NPZ in video directory root
            video_dir = os.path.join(out_dir, video_name)
            data_npz_load = {}
            data_npz_load["coords"] = result["coords"].cpu().numpy()
            # Use full video camera parameters instead of segmented ones
            data_npz_load["extrinsics"] = result["full_extrinsics"].cpu().numpy()
            data_npz_load["intrinsics"] = result["full_intrinsics"].cpu().numpy()
            data_npz_load["height"] = result["video_tensor"].shape[-2]
            data_npz_load["width"] = result["video_tensor"].shape[-1]
            data_npz_load["depths"] = result["depths"].cpu().numpy().astype(np.float16)
            data_npz_load["unc_metric"] = result["depth_conf"].astype(np.float16)
            data_npz_load["visibs"] = result["visibs"][..., None].cpu().numpy()
            if args.save_video:
                data_npz_load["video"] = result["video_tensor"].cpu().numpy()

            save_path = os.path.join(video_dir, video_name + ".npz")
            np.savez(save_path, **data_npz_load)
            logger.info(f"Traditional visualization NPZ saved to {save_path}")

        except Exception as e:
            import traceback

            logger.error(f"Failed to process {video_name}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            continue

    # Cleanup
    del model_3dtracker
    del model_depth_pose
    torch.cuda.empty_cache()
    logger.info("Batch processing completed!")
