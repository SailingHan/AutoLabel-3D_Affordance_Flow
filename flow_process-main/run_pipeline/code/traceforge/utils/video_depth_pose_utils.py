import os
import torch
import numpy as np
from loguru import logger

from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image


def align_depth_scale(pred_depth, known_depth):
    """
    Align the scale of predicted depth to known depth using median scaling.
    Input:
        pred_depth: (H, W), torch tensor
        known_depth: (H, W), torch tensor
    Return:
        aligned_depth: (H, W), torch tensor
    """
    valid_mask = (known_depth > 0) & (pred_depth > 0)
    scale = np.median(known_depth[valid_mask]) / np.median(pred_depth[valid_mask])
    return scale

def align_video_depth_scale(pred_depth, known_depth):
    """
    Align the scale of predicted depth to known depth using median scaling.
    Input:
        pred_depth: (T, H, W), torch tensor
        known_depth: (T, H, W), torch tensor
    Return:
        aligned_depth: (T, H, W), torch tensor
    """
    scales = []
    for t in range(pred_depth.shape[0]):
        scales.append(
            align_depth_scale(pred_depth[t], known_depth[t])
        )
    scale = np.array(scales).mean()
    aligned_depth = pred_depth * scale
    return aligned_depth, scale


class BaseVideoDepthPoseWrapper:
    def __init__(self, args):
        self.args = args
        self.device = args.device

    def __call__(
        self,
        video_tensor,
        known_depth=None,
        known_intrinsics=None,
        stationary_camera=False,
        replace_with_known_depth=True,
        replace_with_known_intrinsics=False,
    ):
        """
        Input:
            video_tensor: (T, 3, H, W), torch tensor, range [0, 1]
            known_depth: (T, H, W), torch tensor, range [0, inf), if provided, will be used as depth input
            known_intrinsics: (3, 3) or (T, 3, 3), optional real camera intrinsics
            stationary_camera: bool, if True, indicates the camera is stationary
            replace_with_known_depth: bool, only used when known_depth is provided. If True, return known_depth as the depth. If False, rescale the estimated depth to the scale of known depth
            replace_with_known_intrinsics: bool, only used when known_intrinsics is provided. If True, replace predicted intrinsics with known intrinsics
        Return:
            video_ten: (T, 3, H, W), torch tensor, range [0, 1], processed (resized) video tensor, will be used in 3D tracking as well
            depth_npy: (T, H, W), numpy array
            depth_conf: (T, H, W), numpy array
            extrs_npy: (T, 4, 4), numpy array, in camera-to-world format
            intrs_npy: (T, 3, 3), numpy array
        """
        raise NotImplementedError


class VGGT4Wrapper(BaseVideoDepthPoseWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.load_model()

    def load_model(self, checkpoint_path="Yuxihenry/SpatialTrackerV2_Front"):
        model = VGGT4Track.from_pretrained(checkpoint_path)
        logger.debug(f"load vggt4 from {checkpoint_path}")
        model = model.eval()
        model = model.to(self.device)
        return model

    def __call__(
        self,
        video_tensor,
        known_depth=None,
        known_intrinsics=None,
        stationary_camera=False,
        replace_with_known_depth=True,
        replace_with_known_intrinsics=False,
    ):
        video_tensor_processed = preprocess_image(video_tensor)[None]  # (1, T, 3, H, W)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                predictions = self.model(video_tensor_processed.cuda())
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = (
                    predictions["points_map"][..., 2],
                    predictions["unc_metric"],
                )

        depth_npy = depth_map.squeeze().cpu().numpy()
        depth_conf_npy = depth_conf.squeeze().cpu().numpy()
        extrs_npy = extrinsic.squeeze().cpu().numpy()
        intrs_npy = intrinsic.squeeze().cpu().numpy()
        video_ten = video_tensor_processed.squeeze()

        if known_depth is not None:
            known_depth = torch.nn.functional.interpolate(
                known_depth[:, None, :, :], size=depth_npy.shape[1:], mode="bilinear", align_corners=False
            )[:, 0, :, :]
            known_depth = known_depth.cpu().numpy()
            depth_npy, scale = align_video_depth_scale(
                depth_npy, known_depth
            )
            if replace_with_known_depth:
                depth_npy = known_depth
                depth_conf_npy = (known_depth > 0).astype(np.float32)

            extrs_npy[:, :3, 3] *= scale

        # Optional real-intrinsics override for fixed-camera datasets.
        # Default behavior remains unchanged unless explicitly enabled.
        if known_intrinsics is not None and replace_with_known_intrinsics:
            if isinstance(known_intrinsics, torch.Tensor):
                known_intrinsics = known_intrinsics.detach().cpu().numpy()
            known_intrinsics = np.asarray(known_intrinsics, dtype=np.float32)
            if known_intrinsics.shape == (3, 3):
                known_intrinsics = np.repeat(known_intrinsics[None], intrs_npy.shape[0], axis=0)
            if known_intrinsics.shape != intrs_npy.shape:
                raise ValueError(
                    f"known_intrinsics must have shape (3,3) or {intrs_npy.shape}, got {known_intrinsics.shape}"
                )
            intrs_npy = known_intrinsics.copy()

        if stationary_camera:
            extrs_npy = np.repeat(extrs_npy[0:1], extrs_npy.shape[0], axis=0)

        return video_ten, depth_npy, depth_conf_npy, extrs_npy, intrs_npy


video_depth_pose_dict = {
    "vggt4": VGGT4Wrapper,
}
