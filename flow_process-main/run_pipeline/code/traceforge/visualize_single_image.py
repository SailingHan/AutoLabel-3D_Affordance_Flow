import os
import sys
import argparse
import time
import numpy as np
import cv2 as cv
from PIL import Image
import viser
import viser.extras
import viser.transforms as tf
TRACEFORGE_ROOT = os.environ.get("TRACEFORGE_ROOT", "/home/zhy/data/TraceForge")
if TRACEFORGE_ROOT and TRACEFORGE_ROOT not in sys.path:
    sys.path.insert(0, TRACEFORGE_ROOT)

from utils.viser_utils import define_track_colors
from utils.threed_utils import unproject_by_depth, inverse_intrinsic, get_meshgrid
from loguru import logger


"""
Usage:
    python visualize_single_image.py \
        --npz_path <output_dir>/<video_name>/samples/<video_name>_<frame>.npz \
        --image_path <output_dir>/<video_name>/images/<video_name>_<frame>.png \
        --depth_path <output_dir>/<video_name>/depth/<video_name>_<frame>.png \
        --port 8080
"""


def load_depth_from_path(depth_path):
    """Load depth data from either PNG or NPZ file"""
    if depth_path.endswith(".npz"):
        depth_data = np.load(depth_path)
        depth = depth_data["depth"]
        depth_data.close()
        return depth
    elif depth_path.endswith(".png"):
        base_path = depth_path[:-4]
        raw_npz_path = f"{base_path}_raw.npz"
        if os.path.exists(raw_npz_path):
            depth_data = np.load(raw_npz_path)
            depth = depth_data["depth"]
            depth_data.close()
            return depth
        depth_img = np.array(Image.open(depth_path))
        depth = depth_img.astype(np.float32) / 10000.0
        return depth
    else:
        raise ValueError(f"Unsupported depth file format: {depth_path}")


def get_camera_params_from_main_npz(episode_dir, frame_idx):
    """Get camera intrinsics and extrinsics from the main NPZ file"""
    episode_name = os.path.basename(episode_dir)
    main_npz_path = os.path.join(episode_dir, f"{episode_name}.npz")

    if os.path.exists(main_npz_path):
        data = np.load(main_npz_path)
        intrinsics = data["intrinsics"][frame_idx]
        extrinsics = data["extrinsics"][frame_idx]
        c2w = np.linalg.inv(extrinsics)
        height, width = int(data["height"]), int(data["width"])
        data.close()
    else:
        print(f"Main NPZ file not found: {main_npz_path}")
        intrinsics = np.array(
            [
                [257.91296, 0.0, 259.0],
                [0.0, 261.4576, 161.0],
                [0.0, 0.0, 1.0],
            ]
        )
        extrinsics = np.array(
            [
                [1.0000000e00, 4.0706014e-05, 8.9567264e-05, 9.0881156e-05],
                [-4.0680898e-05, 9.9999994e-01, -2.8039535e-04, 3.5203320e-05],
                [-8.9578680e-05, 2.8039169e-04, 9.9999994e-01, -2.7687754e-04],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        c2w = np.linalg.inv(extrinsics)
        height, width = 322, 518

    return {
        "K": intrinsics,
        "c2w": c2w,
        "w2c": extrinsics,
        "height": height,
        "width": width,
    }


def convert_image_coords_to_world(traj_image_coords, camera_params):
    n_traj, horizon, _ = traj_image_coords.shape
    k_mat = camera_params["K"]
    c2w = camera_params["c2w"]

    traj_flat = traj_image_coords.reshape(n_traj * horizon, 3)
    world_points = []
    for i in range(n_traj * horizon):
        x, y, z = traj_flat[i]
        x_norm = (x - k_mat[0, 2]) / k_mat[0, 0]
        y_norm = (y - k_mat[1, 2]) / k_mat[1, 1]
        cam_point = np.array([x_norm * z, y_norm * z, z, 1.0])
        world_point = c2w @ cam_point
        world_points.append(world_point[:3])

    world_points = np.array(world_points).reshape(n_traj, horizon, 3)
    return world_points


def visualize_single_image(npz_path, image_path, depth_path, port=8080, max_tracks=80):
    sample_dir = os.path.dirname(npz_path)
    episode_dir = os.path.dirname(sample_dir)
    npz_filename = os.path.basename(npz_path)
    frame_idx = int(npz_filename.split("_")[-1].split(".")[0])

    logger.info(f"Loading data for frame {frame_idx} from {episode_dir}")

    sample_data = np.load(npz_path)
    traj_image_coords = sample_data["traj"]
    keypoints = sample_data["keypoints"]
    valid_steps = sample_data["valid_steps"]
    sample_data.close()

    if max_tracks > 0 and len(traj_image_coords) > max_tracks:
        traj_image_coords = traj_image_coords[:max_tracks]
        keypoints = keypoints[:max_tracks]
    logger.info(
        f"Loaded {len(traj_image_coords)} trajectories with horizon {traj_image_coords.shape[1]}"
    )

    image = np.array(Image.open(image_path)).astype(np.float32) / 255.0
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    depth = load_depth_from_path(depth_path)
    logger.info(f"Image shape: {image.shape}, Depth shape: {depth.shape}")

    camera_params = get_camera_params_from_main_npz(episode_dir, frame_idx)
    traj_world = convert_image_coords_to_world(traj_image_coords, camera_params)

    points_xyz = unproject_by_depth(
        depth=depth[None, None],
        K=camera_params["K"][None],
        c2w=camera_params["c2w"][None],
    )[0].transpose(1, 2, 0)

    downsample_factor = 4
    points_xyz_ds = points_xyz[::downsample_factor, ::downsample_factor].reshape(-1, 3)
    points_rgb_ds = image[::downsample_factor, ::downsample_factor].reshape(-1, 3)

    valid_mask = (points_xyz_ds[:, 2] > 0) & (points_xyz_ds[:, 2] < 10.0)
    points_xyz_ds = points_xyz_ds[valid_mask]
    points_rgb_ds = points_rgb_ds[valid_mask]

    logger.info(f"Point cloud: {len(points_xyz_ds)} points after filtering")

    track_colors = define_track_colors(traj_world, colormap="turbo")

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")
    logger.info(f"Started Viser server at http://localhost:{port}")

    with server.gui.add_folder("Visualization"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.02, step=1e-3, initial_value=0.006
        )
        gui_track_width = server.gui.add_slider(
            "Track width", min=0.5, max=5.0, step=0.5, initial_value=4.0
        )
        gui_track_length = server.gui.add_slider(
            "Track length",
            min=1,
            max=traj_world.shape[1],
            step=1,
            initial_value=min(30, traj_world.shape[1]),
        )

    point_cloud_handle = server.scene.add_point_cloud(
        name="/point_cloud",
        points=points_xyz_ds,
        colors=points_rgb_ds,
        point_size=gui_point_size.value,
    )

    track_handles = []
    start_point_handles = []

    def update_visualization():
        nonlocal track_handles, start_point_handles
        for handle in track_handles:
            handle.remove()
        for handle in start_point_handles:
            handle.remove()
        track_handles = []
        start_point_handles = []

        point_cloud_handle.point_size = gui_point_size.value
        max_track_length = gui_track_length.value

        for i, track in enumerate(traj_world):
            valid_track = track[:max_track_length]
            if len(valid_track) < 2:
                continue

            color = track_colors[i]
            track_handle = server.scene.add_spline_catmull_rom(
                name=f"/track_{i}",
                positions=valid_track,
                color=color,
                line_width=gui_track_width.value,
            )
            track_handles.append(track_handle)

            start_point_handle = server.scene.add_point_cloud(
                name=f"/start_point_{i}",
                points=valid_track[0:1],
                colors=np.array([color]),
                point_size=gui_point_size.value * 3,
            )
            start_point_handles.append(start_point_handle)

    @gui_point_size.on_update
    def _(_event):
        update_visualization()

    @gui_track_width.on_update
    def _(_event):
        update_visualization()

    @gui_track_length.on_update
    def _(_event):
        update_visualization()

    update_visualization()

    logger.info("Visualization ready. Press Ctrl+C to stop the server.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping server...")


def main():
    parser = argparse.ArgumentParser(description="Visualize a single frame 3D scene with trajectories")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to sample NPZ file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to RGB image")
    parser.add_argument("--depth_path", type=str, required=True, help="Path to depth image or NPZ")
    parser.add_argument("--port", type=int, default=8080, help="Port for Viser server")
    parser.add_argument("--max-tracks", type=int, default=80, help="Maximum tracks to draw in Viser; <=0 draws all")
    args = parser.parse_args()

    visualize_single_image(
        npz_path=args.npz_path,
        image_path=args.image_path,
        depth_path=args.depth_path,
        port=args.port,
        max_tracks=args.max_tracks,
    )


if __name__ == "__main__":
    main()
