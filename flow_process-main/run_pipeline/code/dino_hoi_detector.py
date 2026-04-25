#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GroundingDINO-based HOI detector (bbox only).

This is a drop-in replacement for the FasterRCNN-based EgoHOIDetector usage in label_gen_demo.py:
- Returns "hand" boxes (top-K) and "object" boxes (top-1 by default) in XYXY pixel coordinates.
- Boxes are in ORIGINAL image pixel coordinates (before any downsampling). Caller can scale as needed.

Assumptions (matches GroundingDINO util.inference defaults):
- predict() returns:
    boxes:  (N,4) in normalized CXCYWH (0..1)
    logits: (N,)
    phrases: list[str] length N
"""

from __future__ import annotations
import os
import time
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception as e:
    raise ImportError("GroundingDINODetector requires torch.") from e

# GroundingDINO util.inference (from the official repo / pip package)
try:
    from groundingdino.util.inference import load_model, load_image, predict
except Exception as e:
    raise ImportError(
        "Cannot import groundingdino.util.inference. "
        "Make sure GroundingDINO is installed or its repo is in PYTHONPATH."
    ) from e


def _normalize_phrase(s: str) -> str:
    s = s.lower().strip()
    # remove common punctuation from GroundingDINO phrases
    for ch in [".", ",", ";", ":", "!", "?", "\"", "'"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def filter_bbox(image_source, boxes, logits, phrases, m):
    # 过滤box，如果有多个mask为同一个label的，那么就只保留置信度最大的那个
    unique_labels = set(phrases)
    filtered_boxes = []
    filtered_logits = []
    filtered_phrases = []

    for label in unique_labels:
        label_indices = [i for i, phrase in enumerate(phrases) if phrase == label]
        label_logits = logits[label_indices]
        max_logit_index = torch.argmax(label_logits)
        original_index = label_indices[max_logit_index]

        filtered_boxes.append(boxes[original_index])
        filtered_logits.append(logits[original_index])
        filtered_phrases.append(phrases[original_index])

    filtered_boxes = torch.stack(filtered_boxes)
    filtered_logits = torch.stack(filtered_logits)

    # 取置信度最大的前 7 个
    top7_indices = torch.topk(filtered_logits, k=min(m, len(filtered_logits)))[1]
    filtered_boxes = filtered_boxes[top7_indices]
    filtered_logits = filtered_logits[top7_indices]
    filtered_phrases = [filtered_phrases[i] for i in top7_indices]

    print(filtered_phrases)
    print(filtered_logits)

    # 遍历每个过滤后的框，单独保存
    # for i in range(len(filtered_boxes)):
    #     box = filtered_boxes[i]
    #     logit = filtered_logits[i]
    #     phrase = filtered_phrases[i]
        
    #     # 确保参数格式正确
    #     boxes_single = box.unsqueeze(0)  # 形状变为 (1,4)
    #     logits_single = torch.tensor([logit])  # 形状变为 (1,)

    #     phrases_single = [phrase]  # 列表形式
        
    #     # 绘制并保存
    #     annotated_frame = annotate(
    #         image_source=image_source,
    #         boxes=boxes_single,
    #         logits=logits_single,
    #         phrases=phrases_single
    #     )
    #     filename = f"./result/annotated_image_{i}.jpg"  # 避免空格在文件名中
    #     cv2.imwrite(filename, annotated_frame)

    return filtered_boxes, filtered_logits, filtered_phrases



def _cxcywh_to_xyxy_pixel(boxes_cxcywh_norm: torch.Tensor, w: int, h: int) -> np.ndarray:
    """
    boxes_cxcywh_norm: (N,4) tensor in normalized cx,cy,w,h (0..1)
    returns: (N,4) float32 array in pixel xyxy
    """
    if boxes_cxcywh_norm.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)

    b = boxes_cxcywh_norm.detach().cpu().float().numpy()
    cx = b[:, 0] * w
    cy = b[:, 1] * h
    bw = b[:, 2] * w
    bh = b[:, 3] * h
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    out = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)
    # clip
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 2] = np.clip(out[:, 2], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    out[:, 3] = np.clip(out[:, 3], 0, h - 1)
    return out


class GroundingDINODetector:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        box_threshold: float = 0.30,
        text_threshold: float = 0.30,
    ):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

        # Load model ONCE
        self.model = load_model(self.config_path, self.checkpoint_path)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.model.eval()


    def detect(
        self,
        tool_name: str,
        object_name: str,
        image_path: Optional[str] = None,
        image_rgb: Optional[np.ndarray] = None,
        max_tools: int = 1,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Returns:
        {
            "ok": bool,
            "tool_boxes_xyxy": [np.ndarray(4,), ...],   # up to max_tools (default 1)
            "obj_boxes_xyxy":  [np.ndarray(4,), ...],   # 1
            "phrases": [...],   # filtered phrases
            "logits":  [...],

            # backward-compatible aliases:
            "hand_boxes_xyxy":  [...],  # only present when tool_name == 'hand'
        }
        """
        if image_path is None and image_rgb is None:
            raise ValueError("Provide either image_path or image_rgb.")

        import re
        tool_label = tool_name.replace("_", " ").strip().lower()
        obj_label = object_name.replace("_", " ").strip().lower()
        # prompt BOTH tool and object (instead of hard-coding hand)
        text_prompt = f"{tool_label} . {obj_label} ."

        tmp_path = None
        try:
            if image_path is None:
                fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="dino_tmp_")
                os.close(fd)
                import cv2
                bgr = image_rgb[:, :, ::-1].copy()
                cv2.imwrite(tmp_path, bgr)
                img_path = tmp_path
            else:
                img_path = image_path

            image_source, image = load_image(img_path)
            h, w = image_source.shape[:2]

            image = image.to(torch.float32)
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
            # boxes: (N,4) cxcywh normalized (tensor)
            # logits: (N,) (tensor)
            # phrases: list[str]

            # ---------- 你给的 filter_bbox 需要 phrase 精确匹配 ----------
            # DINO 的 phrase 可能是 "hand(0.43)" / "cup(0.51)" / "a hand" 等
            # 我这里先把 phrase 归一化成两个标签： "hand" / obj_label
            def _norm(p: str) -> str:
                p = p.lower()
                p = re.sub(r"\(.*?\)", "", p)     # 去掉 (0.xx)
                p = re.sub(r"[^a-z0-9 ]", " ", p)
                p = " ".join(p.split())
                return p

            tool_words = set(_norm(tool_label).split())
            obj_words = set(_norm(obj_label).split())
            if len(tool_words) == 0:
                tool_words = set([_norm(tool_label)])
            if len(obj_words) == 0:
                obj_words = set([_norm(obj_label)])

            phrases_simple = []
            for p in phrases:
                pn = _norm(p)
                pw = set(pn.split())
                # Prefer tool match first (so "hand" won't swallow the object if tool isn't hand)
                if len(pw & tool_words) > 0:
                    phrases_simple.append(tool_label)
                elif len(pw & obj_words) > 0:
                    phrases_simple.append(obj_label)
                else:
                    # 其它的都丢到 "other"（避免影响你 filter 结果）
                    phrases_simple.append("other")

            # 用你提供的 filter_bbox（不改逻辑）
            filtered_boxes, filtered_logits, filtered_phrases = filter_bbox(
                image_source=image_source,
                boxes=boxes,
                logits=logits,
                phrases=phrases_simple,
                m=7
            )

            # 只要 1 tool + 1 object，不满足就丢
            label2idx = {p: i for i, p in enumerate(filtered_phrases)}
            if (tool_label not in label2idx) or (obj_label not in label2idx):
                return {
                    "ok": False,
                    "tool_boxes_xyxy": [],
                    "obj_boxes_xyxy": [],
                    "hand_boxes_xyxy": [],
                    "phrases": filtered_phrases,
                    "logits": filtered_logits.detach().cpu().tolist() if hasattr(filtered_logits, "detach") else [],
                }

            tool_i = label2idx[tool_label]
            obj_i = label2idx[obj_label]

            # 取出来两个 box（cxcywh normalized）并转成 xyxy pixel
            pick = torch.stack([filtered_boxes[tool_i], filtered_boxes[obj_i]], dim=0)  # (2,4)
            boxes_xyxy = _cxcywh_to_xyxy_pixel(pick, w=w, h=h)  # -> (2,4) numpy float32

            tool_box = boxes_xyxy[0].copy()
            obj_box = boxes_xyxy[1].copy()

            out = {
                "ok": True,
                "tool_boxes_xyxy": [tool_box],  # default 1
                "obj_boxes_xyxy": [obj_box],    # 1
                "phrases": filtered_phrases,
                "logits": filtered_logits.detach().cpu().tolist() if hasattr(filtered_logits, "detach") else [],
            }
            # backward-compatible alias
            out["hand_boxes_xyxy"] = [tool_box] if tool_label == "hand" else []
            return out

        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
