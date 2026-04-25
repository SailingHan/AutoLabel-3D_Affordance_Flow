#!/usr/bin/env python3
"""Adapt BridgeData-style raw episodes into the local TraceForge episode layout.

The adapter is intentionally format-boundary code. Downstream TraceForge code only
sees rvideo_traj_dataset episodes:

  <output>/<task_label>/traj_000/rgb_0.png
  <output>/<task_label>/traj_000/meta.json
  <output>/<task_label>/traj_000/action.npy
  <output>/<task_label>/traj_000/state.npy
  <output>/<task_label>/traj_000/language.txt
  <output>/<task_label>/traj_000/camera_in.npy  (optional)
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from bridge_semantics import generate_semantics, resolve_llm_config
except ImportError:
    from .bridge_semantics import generate_semantics, resolve_llm_config


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_CAMERA_FILENAMES = (
    "camera_in.npy",
    "intrinsics.npy",
    "camera_intrinsics.npy",
    "K.npy",
)


@dataclass(frozen=True)
class BridgeEpisode:
    episode_dir: Path
    image_dir: Path
    pkl_files: tuple[Path, ...]


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def normalize_label(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_+ -]+", "", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "bridge_task"


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_first_key(payloads: list[Any], names: tuple[str, ...]) -> Any | None:
    lowered = {name.lower() for name in names}

    def visit(obj: Any) -> Any | None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).lower() in lowered:
                    return value
            for value in obj.values():
                found = visit(value)
                if found is not None:
                    return found
        if isinstance(obj, (list, tuple)):
            for value in obj:
                found = visit(value)
                if found is not None:
                    return found
        return None

    for payload in payloads:
        found = visit(payload)
        if found is not None:
            return found
    return None


def to_array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype == object:
        try:
            arr = np.stack(value)
        except Exception:
            return None
    if arr.size == 0:
        return None
    if not np.issubdtype(arr.dtype, np.number) and arr.dtype != bool:
        return None
    return arr


def find_stacked_list_key(payloads: list[Any], key_name: str) -> np.ndarray | None:
    key_l = key_name.lower()

    def visit(obj: Any) -> np.ndarray | None:
        if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, dict) for x in obj):
            vals = []
            for item in obj:
                for key, value in item.items():
                    if str(key).lower() == key_l:
                        arr = to_array_or_none(value)
                        if arr is not None:
                            vals.append(arr)
                        break
            if vals:
                try:
                    return np.stack(vals)
                except Exception:
                    return np.asarray(vals)
        if isinstance(obj, dict):
            for value in obj.values():
                found = visit(value)
                if found is not None:
                    return found
        if isinstance(obj, (list, tuple)):
            for value in obj:
                found = visit(value)
                if found is not None:
                    return found
        return None

    for payload in payloads:
        found = visit(payload)
        if found is not None:
            return found
    return None


def infer_task_text_from_path(episode_dir: Path, task_label: str) -> str:
    parts = list(episode_dir.parts)
    if "scripted_raw" in parts:
        idx = parts.index("scripted_raw")
        if idx + 1 < len(parts):
            return re.sub(r"_\d+-\d+$", "", parts[idx + 1]).replace("_", " ")
    if task_label:
        return task_label.replace("_", " ")
    return episode_dir.parent.name.replace("_", " ")


def find_language(payloads: list[Any], episode_dir: Path, task_label: str = "") -> str:
    value = find_first_key(
        payloads,
        (
            "language",
            "language_instruction",
            "instruction",
            "task",
            "task_description",
            "natural_language",
        ),
    )
    if value is not None:
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        text = str(value).strip()
        if text:
            return text
    return infer_task_text_from_path(episode_dir, task_label)


def find_image_dir(root: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        count = sum(1 for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTS)
        if count > 0:
            bonus = 1000 if path.name.lower() in {"images0", "images", "rgb", "frames"} else 0
            candidates.append((count + bonus, path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (-x[0], len(x[1].parts), str(x[1])))[0][1]


def discover_episodes(raw_root: Path) -> list[BridgeEpisode]:
    episodes: list[BridgeEpisode] = []
    seen: set[Path] = set()
    candidate_dirs = [raw_root] + [p for p in raw_root.rglob("*") if p.is_dir()]
    for image_dir in sorted(candidate_dirs, key=lambda p: str(p)):
        image_files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not image_files:
            continue
        episode_dir = image_dir.parent if image_dir.name.lower() in {"images0", "images", "rgb", "frames"} else image_dir
        if episode_dir in seen:
            continue
        seen.add(episode_dir)
        pkl_files = tuple(sorted(episode_dir.rglob("*.pkl"), key=natural_key))
        episodes.append(BridgeEpisode(episode_dir=episode_dir, image_dir=image_dir, pkl_files=pkl_files))
    return episodes


def copy_images(image_dir: Path, out_dir: Path, max_frames: int) -> int:
    files = sorted(
        (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=natural_key,
    )
    if max_frames > 0:
        files = files[:max_frames]
    for idx, src in enumerate(files):
        dst = out_dir / f"rgb_{idx}.png"
        if src.suffix.lower() == ".png":
            shutil.copy2(src, dst)
        else:
            Image.open(src).convert("RGB").save(dst)
    return len(files)


def copy_camera_intrinsics(episode: BridgeEpisode, out_dir: Path) -> str | None:
    for name in DEFAULT_CAMERA_FILENAMES:
        for src in (episode.episode_dir / name, episode.image_dir / name):
            if src.exists():
                shutil.copy2(src, out_dir / "camera_in.npy")
                return str(src)
    return None


def adapt_episode(
    episode: BridgeEpisode,
    out_dir: Path,
    traj_name: str,
    task_label: str,
    default_tool: str,
    default_target: str,
    max_frames: int,
    overwrite: bool,
    semantic_mode: str,
    semantic_provider: str,
    semantic_model: str,
    semantic_base_url: str,
    semantic_api_key_env: str,
    semantic_timeout_sec: int,
    semantic_retries: int,
) -> Path | None:
    traj_dir = out_dir / task_label / traj_name
    if traj_dir.exists() and not overwrite:
        return traj_dir
    traj_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for stale in traj_dir.glob("rgb_*.png"):
            stale.unlink()
        for stale in traj_dir.glob("depth_*.png"):
            stale.unlink()

    frame_count = copy_images(episode.image_dir, traj_dir, max_frames=max_frames)
    if frame_count == 0:
        return None

    payloads = []
    loaded_pkl = []
    for pkl_path in episode.pkl_files:
        try:
            payload = load_pickle(pkl_path)
        except Exception as exc:
            loaded_pkl.append({"path": str(pkl_path), "error": str(exc)})
            continue
        payloads.append(payload)
        loaded_pkl.append({"path": str(pkl_path), "summary": safe_json(payload)})

    language = find_language(payloads, episode.episode_dir, task_label=task_label)
    action = find_stacked_list_key(payloads, "actions")
    if action is None:
        action = to_array_or_none(find_first_key(payloads, ("actions", "action", "act")))
    state = to_array_or_none(find_first_key(payloads, ("states", "state", "robot_state", "proprio", "proprioception")))
    if action is None:
        action = np.zeros((0,), dtype=np.float32)
    if state is None:
        state = np.zeros((0,), dtype=np.float32)
    np.save(traj_dir / "action.npy", action)
    np.save(traj_dir / "state.npy", state)

    camera_source = copy_camera_intrinsics(episode, traj_dir)
    with open(traj_dir / "language.txt", "w", encoding="utf-8") as f:
        f.write(language.strip() + "\n")

    meta_for_semantics = {
        "task_label": task_label,
        "action_shape": list(action.shape),
        "state_shape": list(state.shape),
        "default_tool": default_tool,
        "default_target": default_target,
    }
    semantic_candidate_path = traj_dir / "semantic_candidates.json"
    semantic_candidate_path.write_text(json.dumps(meta_for_semantics, indent=2), encoding="utf-8")
    semantics, semantic_debug = generate_semantics(
        episode_dir=traj_dir,
        task_name=task_label,
        language=language,
        meta=meta_for_semantics,
        mode=semantic_mode,
        provider=semantic_provider,
        model=semantic_model,
        base_url=semantic_base_url,
        api_key_env=semantic_api_key_env,
        timeout_sec=semantic_timeout_sec,
        retries=semantic_retries,
    )
    llm_semantics = {
        "source_task": task_label,
        "sample_id": traj_name,
        "sample_dir": str(episode.episode_dir),
        "analysis": {"folder_label_norm": task_label},
        "llm_result": semantics,
        "semantic_debug": semantic_debug,
        "adapter_note": "BridgeData semantics are generated from rule candidates plus optional LLM standardization.",
    }
    with open(traj_dir / "llm_semantics.json", "w", encoding="utf-8") as f:
        json.dump(llm_semantics, f, indent=2)

    meta = {
        "source": "bridgedata_v2_raw",
        "source_episode_dir": str(episode.episode_dir),
        "source_image_dir": str(episode.image_dir),
        "frame_count": frame_count,
        "language": language,
        "task_label": task_label,
        "action_shape": list(action.shape),
        "state_shape": list(state.shape),
        "camera_in_source": camera_source,
        "pkl_files": loaded_pkl,
    }
    with open(traj_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return traj_dir


def main() -> int:
    parser = argparse.ArgumentParser("Adapt BridgeData raw JPEG/PNG/pkl episodes into rvideo_traj_dataset layout")
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-label", type=str, default="")
    parser.add_argument("--default-tool", type=str, default="hand")
    parser.add_argument("--default-target", type=str, default="object")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--semantic-mode", choices=["rules", "rules_then_llm", "llm"], default="llm")
    parser.add_argument("--semantic-provider", choices=["custom", "moonshot", "openai"], default="moonshot")
    parser.add_argument("--semantic-model", type=str, default=None)
    parser.add_argument("--semantic-base-url", type=str, default=None)
    parser.add_argument("--semantic-api-key-env", type=str, default=None)
    parser.add_argument("--semantic-timeout-sec", type=int, default=120)
    parser.add_argument("--semantic-retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    llm_config = resolve_llm_config(
        args.semantic_provider,
        args.semantic_model,
        args.semantic_base_url,
        args.semantic_api_key_env,
    )

    raw_root = args.bridge_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    episodes = discover_episodes(raw_root)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise SystemExit(f"No BridgeData episodes with JPEG/PNG frames found under {raw_root}")

    manifest = []
    manifest_path = output_root / "bridge_adapter_manifest.json"
    counters: dict[str, int] = {}
    for episode in episodes:
        task_label = normalize_label(args.task_label or episode.episode_dir.parent.name)
        idx = counters.get(task_label, 0)
        counters[task_label] = idx + 1
        print(
            f"[bridge_adapter] adapting {idx + 1}/{len(episodes)} "
            f"task={task_label} source={episode.episode_dir}",
            flush=True,
        )
        traj_dir = adapt_episode(
            episode=episode,
            out_dir=output_root,
            traj_name=f"traj_{idx:06d}",
            task_label=task_label,
            default_tool=args.default_tool,
            default_target=args.default_target,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
            semantic_mode=args.semantic_mode,
            semantic_provider=llm_config["provider"],
            semantic_model=llm_config["model"],
            semantic_base_url=llm_config["base_url"],
            semantic_api_key_env=llm_config["api_key_env"],
            semantic_timeout_sec=args.semantic_timeout_sec,
            semantic_retries=args.semantic_retries,
        )
        if traj_dir is not None:
            manifest.append({"source_episode_dir": str(episode.episode_dir), "traj_dir": str(traj_dir)})
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"bridge_root": str(raw_root), "episodes": manifest}, f, indent=2)
            print(f"[bridge_adapter] wrote {traj_dir}", flush=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"bridge_root": str(raw_root), "episodes": manifest}, f, indent=2)
    print(f"[bridge_adapter] exported {len(manifest)} episodes to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
