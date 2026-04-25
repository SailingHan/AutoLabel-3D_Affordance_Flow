import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def parse_task_label(task_label: str) -> tuple[str, str, str]:
    parts = str(task_label).strip().split("_")
    action = parts[0] if parts else "unknown"
    if "to" in parts:
        to_idx = parts.index("to")
        tool = parts[1] if to_idx >= 2 else "hand"
        target = "_".join(parts[to_idx + 1:]) if (to_idx + 1) < len(parts) else "object"
    else:
        tool = "hand"
        target = "_".join(parts[1:]) if len(parts) > 1 else "object"
    return action, tool, target


def load_role_prompts(traj_dir: str) -> dict:
    traj_path = Path(traj_dir)
    llm_path = traj_path / "llm_semantics.json"
    if llm_path.exists():
        with open(llm_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        llm_result = payload.get("llm_result", {})
        tool = str(llm_result.get("tool", "hand")).strip().lower() or "hand"
        target = str(llm_result.get("target", "object")).strip().lower() or "object"
        action = str(llm_result.get("action", "unknown")).strip().lower() or "unknown"
        return {
            "action": action,
            "tool": tool.replace(" ", "_"),
            "target": target.replace(" ", "_"),
            "source": str(llm_path),
        }

    task_label = traj_path.parent.name
    action, tool, target = parse_task_label(task_label)
    return {
        "action": action,
        "tool": tool,
        "target": target,
        "source": f"task_dir:{task_label}",
    }


def sample_mask_points(mask: np.ndarray, n_points: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or n_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    coords = np.stack([xs, ys], axis=1)
    if coords.shape[0] <= n_points:
        return coords.astype(np.float32)
    picks = np.linspace(0, coords.shape[0] - 1, num=n_points, dtype=np.int64)
    return coords[picks].astype(np.float32)


def dilate_mask(mask: np.ndarray | None, radius: int) -> np.ndarray | None:
    if mask is None:
        return None
    out = (mask > 0).astype(np.uint8)
    if radius <= 0:
        return out
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(out, kernel, iterations=1)


@dataclass
class RoleQueryResult:
    query_points: np.ndarray
    query_role_ids: np.ndarray
    role_id_to_name: dict[int, str]
    debug_masks: dict[str, np.ndarray]
    prompts: dict


class RoleQuerySampler:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.enabled = str(getattr(args, "query_mode", "grid")).strip().lower() == "masked_roles"
        self.sam2_image_predictor = None
        self.dino_detector = None
        if not self.enabled:
            return

        local_code_root = str(Path(__file__).resolve().parents[2])
        rvideo_root = str(getattr(args, "rvideo_pipeline_root", "/home/zhy/data/RVideo/pipeline_current"))
        sam2_root = str(getattr(args, "sam2_repo_root", "/home/zhy/data/sam2"))
        dino_root = str(getattr(args, "groundingdino_repo_root", "/home/zhy/data/GroundingDINO"))
        for repo_root in (local_code_root, rvideo_root, sam2_root, dino_root):
            if repo_root and repo_root not in sys.path:
                sys.path.insert(0, repo_root)

        from dino_hoi_detector import GroundingDINODetector
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_cfg = self._normalize_sam2_config(str(args.sam2_cfg), sam2_root)
        sam2_model = build_sam2(sam2_cfg, args.sam2_ckpt, device=self.device)
        self.sam2_image_predictor = SAM2ImagePredictor(sam2_model)
        self.dino_detector = GroundingDINODetector(
            config_path=args.dino_cfg,
            checkpoint_path=args.dino_ckpt,
            device=self.device,
            box_threshold=float(getattr(args, "dino_box_thresh", 0.30)),
            text_threshold=float(getattr(args, "dino_text_thresh", 0.30)),
        )

    @staticmethod
    def _normalize_sam2_config(config_value: str, sam2_root: str) -> str:
        config_value = str(config_value).strip()
        if config_value.startswith("configs/"):
            return config_value
        sam2_pkg_root = Path(sam2_root) / "sam2"
        try:
            cfg_path = Path(config_value).resolve()
            if cfg_path.is_file():
                try:
                    rel_path = cfg_path.relative_to(sam2_pkg_root.resolve())
                    return rel_path.as_posix()
                except ValueError:
                    pass
        except Exception:
            pass
        return config_value

    def _sam2_mask_from_xyxy(self, image_rgb: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray | None:
        image_u8 = np.ascontiguousarray(image_rgb.astype(np.uint8))
        self.sam2_image_predictor.set_image(image_u8)
        box = np.asarray(box_xyxy, dtype=np.float32)[None, :]
        masks, *_ = self.sam2_image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks.ndim == 3:
            mask = masks[0]
        elif masks.ndim == 2:
            mask = masks
        else:
            return None
        mask = (mask > 0).astype(np.uint8)
        if mask.sum() == 0:
            return None
        return mask

    def _segment_pair(
        self,
        image_rgb: np.ndarray,
        image_path: str,
        tool_name: str,
        object_name: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        det = self.dino_detector.detect(
            tool_name=tool_name,
            object_name=object_name,
            image_rgb=image_rgb,
            image_path=image_path,
            max_tools=1,
        )
        tool_boxes = det.get("tool_boxes_xyxy", [])
        obj_boxes = det.get("obj_boxes_xyxy", [])
        if not tool_boxes or not obj_boxes:
            return None, None
        tool_mask = self._sam2_mask_from_xyxy(image_rgb, tool_boxes[0])
        obj_mask = self._sam2_mask_from_xyxy(image_rgb, obj_boxes[0])
        return tool_mask, obj_mask

    def sample_queries(
        self,
        *,
        frame_idx: int,
        image_rgb: np.ndarray,
        image_path: str,
        traj_dir: str,
        frame_h: int,
        frame_w: int,
    ) -> RoleQueryResult:
        prompts = load_role_prompts(traj_dir)
        tool_name = prompts["tool"].replace("_", " ")
        target_name = prompts["target"].replace("_", " ").replace("-", " ")
        total_points = int(getattr(self.args, "masked_query_points", 400))
        dilate_px = int(getattr(self.args, "query_mask_dilate_px", 5))

        role_to_points = {}
        debug_masks = {}
        role_id_to_name = {}

        if prompts["tool"] == "hand":
            hand_mask, target_mask = self._segment_pair(
                image_rgb=image_rgb,
                image_path=image_path,
                tool_name="hand",
                object_name=target_name,
            )
            hand_mask = dilate_mask(hand_mask, dilate_px)
            target_mask = dilate_mask(target_mask, dilate_px)
            if hand_mask is None or target_mask is None:
                raise RuntimeError(
                    f"masked_roles requires both hand and target masks, got hand={hand_mask is not None}, "
                    f"target={target_mask is not None} for prompts {prompts}"
                )
            debug_masks["hand"] = hand_mask if hand_mask is not None else np.zeros((frame_h, frame_w), dtype=np.uint8)
            debug_masks["target"] = target_mask if target_mask is not None else np.zeros((frame_h, frame_w), dtype=np.uint8)
            allocations = [("hand", hand_mask, int(round(total_points * 0.45))), ("target", target_mask, total_points)]
        else:
            tool_mask, target_mask = self._segment_pair(
                image_rgb=image_rgb,
                image_path=image_path,
                tool_name=tool_name,
                object_name=target_name,
            )
            hand_mask, _ = self._segment_pair(
                image_rgb=image_rgb,
                image_path=image_path,
                tool_name="hand",
                object_name=tool_name,
            )
            hand_mask = dilate_mask(hand_mask, dilate_px)
            tool_mask = dilate_mask(tool_mask, dilate_px)
            target_mask = dilate_mask(target_mask, dilate_px)
            if tool_mask is None or target_mask is None:
                raise RuntimeError(
                    f"masked_roles requires both tool and target masks, got tool={tool_mask is not None}, "
                    f"target={target_mask is not None} for prompts {prompts}"
                )
            debug_masks["hand"] = hand_mask if hand_mask is not None else np.zeros((frame_h, frame_w), dtype=np.uint8)
            debug_masks["tool"] = tool_mask if tool_mask is not None else np.zeros((frame_h, frame_w), dtype=np.uint8)
            debug_masks["target"] = target_mask if target_mask is not None else np.zeros((frame_h, frame_w), dtype=np.uint8)
            allocations = [
                ("hand", hand_mask, int(round(total_points * 0.25))),
                ("tool", tool_mask, int(round(total_points * 0.35))),
                ("target", target_mask, total_points),
            ]

        assigned = 0
        active_allocations = []
        for role_name, mask, budget in allocations:
            if mask is None or int(mask.sum()) == 0:
                continue
            active_allocations.append((role_name, mask, max(1, budget)))
        if not active_allocations:
            raise RuntimeError(
                f"masked_roles query generation failed for {traj_dir} frame {frame_idx}: "
                f"no valid hand/tool/target masks from prompts {prompts}"
            )

        for role_idx, (role_name, mask, budget) in enumerate(active_allocations, start=1):
            role_id_to_name[role_idx] = role_name
            if role_idx == len(active_allocations):
                n_points = max(1, total_points - assigned)
            else:
                n_points = min(max(1, budget), max(1, total_points - assigned - (len(active_allocations) - role_idx)))
            assigned += n_points
            role_to_points[role_idx] = sample_mask_points(mask, n_points)

        query_points = []
        query_role_ids = []
        for role_idx, points_xy in role_to_points.items():
            if points_xy.shape[0] == 0:
                continue
            frame_col = np.full((points_xy.shape[0], 1), float(frame_idx), dtype=np.float32)
            query_points.append(np.concatenate([frame_col, points_xy], axis=1))
            query_role_ids.append(np.full((points_xy.shape[0],), role_idx, dtype=np.int32))

        if not query_points:
            raise RuntimeError(f"no masked query points sampled for {traj_dir} frame {frame_idx}")

        return RoleQueryResult(
            query_points=np.concatenate(query_points, axis=0).astype(np.float32),
            query_role_ids=np.concatenate(query_role_ids, axis=0).astype(np.int32),
            role_id_to_name=role_id_to_name,
            debug_masks=debug_masks,
            prompts=prompts,
        )
