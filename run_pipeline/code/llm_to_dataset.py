#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import requests


LLM_PROVIDER_PRESETS = {
    "moonshot": {
        "model": "kimi-k2.5",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "openai": {
        "model": None,
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "custom": {
        "model": None,
        "base_url": None,
        "api_key_env": None,
    },
}


def resolve_llm_config(provider: str, model: str | None, base_url: str | None, api_key_env: str | None) -> dict[str, str]:
    preset = LLM_PROVIDER_PRESETS[provider]
    resolved = {
        "provider": provider,
        "model": model or preset.get("model"),
        "base_url": base_url or preset.get("base_url"),
        "api_key_env": api_key_env or preset.get("api_key_env"),
    }
    missing = [k for k in ("model", "base_url", "api_key_env") if not resolved.get(k)]
    if missing:
        raise RuntimeError(
            f"missing LLM config fields for provider={provider}: {', '.join(missing)}. "
            "Pass --model/--base_url/--api_key_env explicitly or use a preset provider."
        )
    return {k: str(v) for k, v in resolved.items()}


DEFAULT_PROMPT = """You are analyzing short first-person manipulation videos from an RGB-D dataset.

Your job:
1. Describe the video briefly.
2. Identify the core manipulation action with the shortest correct verb phrase.
3. Identify the primary tool or active manipulator.
4. Identify the primary target object being acted on.
5. Describe the interaction between tool and target in one short phrase.
6. Produce a compact action-folder style label.
7. Keep labels concrete and physically grounded in what is visible.

Important normalization rules:
- If the human hand is the manipulator, set tool to "hand".
- Use short lowercase noun phrases.
- Prefer action verbs like pick_up, place, cut, open, close, hang, pour, insert, remove, push, pull, rotate.
- The description must be short, ideally 3 to 8 words, and should omit scene clutter.
- The folder label must be concise and directly usable as an action directory name.
- Prefer a label format like action_target when tool is hand, or action_tool_to_target when the tool matters.
- Do not include uncertain hedging words like maybe, likely, probably.
- If uncertain, still give your best guess and lower confidence.
- Focus on the dominant action in the clip, not incidental motion.
"""


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_+ -]+", "", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def jpeg_data_url(image_bgr, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("failed to encode frame as JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def sample_video_frames(video_path: Path, num_frames: int) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((idx, frame))
            idx += 1
        cap.release()
        frame_count = len(frames)
        if frame_count == 0:
            raise RuntimeError(f"video has no frames: {video_path}")
        picks = evenly_spaced_indices(frame_count, num_frames)
        return [
            {
                "frame_index": frames[i][0],
                "timestamp_sec": (frames[i][0] / fps) if fps > 0 else None,
                "image_url": jpeg_data_url(frames[i][1]),
            }
            for i in picks
        ]

    picks = evenly_spaced_indices(frame_count, num_frames)
    sampled = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        sampled.append(
            {
                "frame_index": int(idx),
                "timestamp_sec": (float(idx) / fps) if fps > 0 else None,
                "image_url": jpeg_data_url(frame),
            }
        )
    cap.release()
    if not sampled:
        raise RuntimeError(f"failed to sample any frames from {video_path}")
    return sampled


def extract_all_video_frames(video_path: Path, prefix: str, output_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = output_dir / f"{prefix}_{count}.png"
        if not cv2.imwrite(str(out_path), frame):
            cap.release()
            raise RuntimeError(f"failed to write frame: {out_path}")
        count += 1
    cap.release()
    return count


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def copy_depth_raw_frames(sample_dir: Path, output_dir: Path) -> int:
    depth_raw_dir = sample_dir / "depth_raw"
    if not depth_raw_dir.exists():
        return 0
    files = sorted(depth_raw_dir.glob("*.png"), key=natural_key)
    for idx, src in enumerate(files):
        shutil.copy2(src, output_dir / f"depth_{idx}.png")
    return len(files)


def evenly_spaced_indices(total: int, want: int) -> list[int]:
    want = max(1, min(int(want), int(total)))
    if want == 1:
        return [0]
    return sorted({int(round(i * (total - 1) / (want - 1))) for i in range(want)})


def build_response_payload(model: str, prompt: str, task_name: str, sample_id: str, frames: list[dict[str, Any]]) -> dict[str, Any]:
    content = [
        {
            "type": "text",
            "text": (
                f"{prompt}\n\n"
                f"Dataset task bucket: {task_name}\n"
                f"Sample id: {sample_id}\n"
                f"You are given several frames sampled across the whole video.\n"
                f"Return strict JSON only, with no markdown fences and no extra commentary."
            ),
        }
    ]
    for frame in frames:
        ts = frame["timestamp_sec"]
        tag = f"frame_index={frame['frame_index']}"
        if ts is not None:
            tag += f", timestamp_sec={ts:.3f}"
        content.append({"type": "text", "text": tag})
        content.append({"type": "image_url", "image_url": {"url": frame["image_url"]}})

    json_contract = {
        "core_description": "short string",
        "action": "short verb phrase",
        "tool": "short noun phrase",
        "target": "short noun phrase",
        "interaction": "short phrase describing tool-target interaction",
        "folder_label": "compact action-folder label",
        "scene_objects": ["list", "of", "visible", "objects"],
        "reasoning_brief": "very short explanation",
        "confidence": 0.0,
    }
    content.append(
        {
            "type": "text",
            "text": "Use exactly this JSON shape:\n" + json.dumps(json_contract, ensure_ascii=False, indent=2),
        }
    )

    return {"model": model, "messages": [{"role": "user", "content": content}]}


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"failed to find JSON object in model output: {text[:1000]}")
    return json.loads(match.group(0))


def call_chat_completions(api_key: str, base_url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"missing choices in response: {json.dumps(data)[:1000]}")
    message = choices[0].get("message") or {}
    text = message.get("content")
    if not text:
        raise RuntimeError(f"missing message.content in response: {json.dumps(data)[:1000]}")
    parsed = extract_json_object(text)
    parsed["_raw_response_id"] = data.get("id")
    parsed["_model"] = data.get("model")
    return parsed


def collect_video_samples(video_root: Path) -> list[dict[str, Any]]:
    samples = []
    for task_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
        nested_dirs = sorted(p for p in task_dir.rglob("*") if p.is_dir() and (p / "rgb.mp4").exists())
        for sample_dir in nested_dirs:
            sample_id = sample_dir.name
            meta_path = sample_dir / "meta.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            samples.append(
                {
                    "task_name": task_dir.name,
                    "sample_id": sample_id,
                    "sample_dir": str(sample_dir),
                    "rgb_video": str(sample_dir / "rgb.mp4"),
                    "meta": meta,
                }
            )
    return samples


def summarize_task_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    task_to_rows = defaultdict(list)
    for row in results:
        task_to_rows[row["task_name"]].append(row)

    summaries = {}
    for task_name, rows in sorted(task_to_rows.items()):
        label_counter = Counter(r["analysis"]["folder_label_norm"] for r in rows)
        interaction_counter = Counter(r["analysis"]["interaction_norm"] for r in rows)
        action_counter = Counter(r["analysis"]["action_norm"] for r in rows)
        tool_counter = Counter(r["analysis"]["tool_norm"] for r in rows)
        target_counter = Counter(r["analysis"]["target_norm"] for r in rows)
        desc_counter = Counter(r["analysis"]["core_description"] for r in rows)
        avg_conf = sum(float(r["analysis"]["confidence"]) for r in rows) / max(1, len(rows))

        summaries[task_name] = {
            "num_samples": len(rows),
            "top_folder_labels": label_counter.most_common(5),
            "top_interactions": interaction_counter.most_common(5),
            "top_actions": action_counter.most_common(5),
            "top_tools": tool_counter.most_common(5),
            "top_targets": target_counter.most_common(5),
            "top_descriptions": desc_counter.most_common(5),
            "avg_confidence": avg_conf,
        }
    return summaries


def export_dataset_like_layout(output_dir: Path, results: list[dict[str, Any]], overwrite: bool) -> Path:
    dataset_root = output_dir
    dataset_root.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in results:
        label = row["analysis"]["folder_label_norm"] or "unknown_action"
        grouped[label].append(row)

    for action_label, rows in sorted(grouped.items()):
        action_dir = dataset_root / action_label
        action_dir.mkdir(parents=True, exist_ok=True)

        action_summary = {
            "action_folder": action_label,
            "num_trajs": len(rows),
            "source_tasks": Counter(r["task_name"] for r in rows).most_common(),
            "top_interactions": Counter(r["analysis"]["interaction_norm"] for r in rows).most_common(5),
            "top_descriptions": Counter(r["analysis"]["core_description"] for r in rows).most_common(5),
        }
        with open(action_dir / "action_info.json", "w", encoding="utf-8") as f:
            json.dump(action_summary, f, ensure_ascii=False, indent=2)

        for traj_idx, row in enumerate(rows):
            traj_dir = action_dir / f"traj_{traj_idx:03d}"
            if traj_dir.exists() and overwrite:
                shutil.rmtree(traj_dir)
            traj_dir.mkdir(parents=True, exist_ok=True)

            sample_src_dir = Path(row["sample_dir"])
            rgb_count = extract_all_video_frames(Path(row["rgb_video"]), "rgb", traj_dir)
            depth_color_count = extract_all_video_frames(sample_src_dir / "depth_vis.mp4", "depth_color", traj_dir)
            depth_count = copy_depth_raw_frames(sample_src_dir, traj_dir)

            camera_src = sample_src_dir / "camera_in.npy"
            if camera_src.exists():
                shutil.copy2(camera_src, traj_dir / "camera_in.npy")

            traj_payload = {
                "source_task": row["task_name"],
                "source_sample_id": row["sample_id"],
                "source_sample_dir": row["sample_dir"],
                "rgb_video": row["rgb_video"],
                "frames_used_for_llm": row["frames_used"],
                "export_counts": {
                    "rgb": rgb_count,
                    "depth": depth_count,
                    "depth_color": depth_color_count,
                },
                "meta": row["meta"],
                "llm_result": {
                    "core_description": row["analysis"]["core_description"],
                    "action": row["analysis"]["action"],
                    "tool": row["analysis"]["tool"],
                    "target": row["analysis"]["target"],
                    "interaction": row["analysis"]["interaction"],
                    "folder_label": row["analysis"]["folder_label"],
                    "confidence": row["analysis"]["confidence"],
                    "scene_objects": row["analysis"]["scene_objects"],
                },
            }
            with open(traj_dir / "llm_semantics.json", "w", encoding="utf-8") as f:
                json.dump(traj_payload, f, ensure_ascii=False, indent=2)

    return dataset_root

def main() -> int:
    ap = argparse.ArgumentParser(
        "Analyze datasets/video tasks with a multimodal API; let the LLM directly produce description, tool, target, interaction, and action folder label."
    )
    ap.add_argument("--video_root", type=Path, default=Path("/home/zhy/data/datasets/video"))
    ap.add_argument("--output_dir", type=Path, default=Path("/home/zhy/data/datasets"))
    ap.add_argument("--llm_provider", choices=sorted(LLM_PROVIDER_PRESETS), default="moonshot")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--base_url", type=str, default=None)
    ap.add_argument("--frames_per_video", type=int, default=6)
    ap.add_argument("--max_samples_per_task", type=int, default=0, help="0 means all samples")
    ap.add_argument("--task_filter", type=str, default="", help="comma-separated task names, e.g. task1,task3")
    ap.add_argument("--api_key_env", type=str, default=None)
    ap.add_argument("--timeout_sec", type=int, default=180)
    ap.add_argument("--sleep_sec", type=float, default=0.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    llm_config = resolve_llm_config(args.llm_provider, args.model, args.base_url, args.api_key_env)

    api_key = os.environ.get(llm_config["api_key_env"], "")
    if not api_key:
        print(f"missing API key env: {llm_config['api_key_env']}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = args.output_dir / "_video_llm_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    requested_tasks = {t.strip() for t in args.task_filter.split(",") if t.strip()}
    samples = collect_video_samples(args.video_root)
    if requested_tasks:
        samples = [s for s in samples if s["task_name"] in requested_tasks]

    per_task_counter = Counter()
    filtered_samples = []
    for sample in samples:
        if args.max_samples_per_task > 0 and per_task_counter[sample["task_name"]] >= args.max_samples_per_task:
            continue
        filtered_samples.append(sample)
        per_task_counter[sample["task_name"]] += 1
    samples = filtered_samples

    results_path = analysis_dir / "video_llm_samples.jsonl"
    existing_done = set()
    if results_path.exists() and not args.overwrite:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                existing_done.add((row["task_name"], row["sample_id"]))

    results = []
    if results_path.exists() and not args.overwrite:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    elif args.overwrite and results_path.exists():
        results_path.unlink()

    with open(results_path, "a", encoding="utf-8") as out_f:
        for idx, sample in enumerate(samples, start=1):
            key = (sample["task_name"], sample["sample_id"])
            if key in existing_done:
                continue

            frames = sample_video_frames(Path(sample["rgb_video"]), args.frames_per_video)
            payload = build_response_payload(
                model=llm_config["model"],
                prompt=DEFAULT_PROMPT,
                task_name=sample["task_name"],
                sample_id=sample["sample_id"],
                frames=frames,
            )
            analysis = call_chat_completions(api_key, llm_config["base_url"], payload, args.timeout_sec)
            analysis["action_norm"] = normalize_text(analysis["action"])
            analysis["tool_norm"] = normalize_text(analysis["tool"])
            analysis["target_norm"] = normalize_text(analysis["target"])
            analysis["interaction_norm"] = normalize_text(analysis["interaction"])
            analysis["folder_label_norm"] = normalize_text(analysis["folder_label"])
            analysis["core_description"] = " ".join(str(analysis["core_description"]).strip().split())
            analysis["interaction"] = " ".join(str(analysis["interaction"]).strip().split())
            analysis["folder_label"] = str(analysis["folder_label"]).strip()

            row = {
                "task_name": sample["task_name"],
                "sample_id": sample["sample_id"],
                "sample_dir": sample["sample_dir"],
                "rgb_video": sample["rgb_video"],
                "meta": sample["meta"],
                "frames_used": [
                    {
                        "frame_index": f["frame_index"],
                        "timestamp_sec": f["timestamp_sec"],
                    }
                    for f in frames
                ],
                "analysis": analysis,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            results.append(row)
            print(
                f"[{idx}/{len(samples)}] {sample['task_name']}/{sample['sample_id']} -> "
                f"{analysis['folder_label_norm']} "
                f"({analysis['action_norm']} | {analysis['tool_norm']} | {analysis['target_norm']})"
            )
            if args.sleep_sec > 0:
                time.sleep(float(args.sleep_sec))

    task_summaries = summarize_task_results(results)
    dataset_root = export_dataset_like_layout(args.output_dir, results, args.overwrite)

    with open(analysis_dir / "video_llm_task_summary.json", "w", encoding="utf-8") as f:
        json.dump(task_summaries, f, ensure_ascii=False, indent=2)

    manifest = {
        "video_root": str(args.video_root),
        "llm_provider": llm_config["provider"],
        "model": llm_config["model"],
        "base_url": llm_config["base_url"],
        "api_key_env": llm_config["api_key_env"],
        "frames_per_video": args.frames_per_video,
        "max_samples_per_task": args.max_samples_per_task,
        "num_results": len(results),
        "task_filter": sorted(requested_tasks),
    }
    with open(analysis_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] sample results: {results_path}")
    print(f"[DONE] task summary: {analysis_dir / 'video_llm_task_summary.json'}")
    print(f"[DONE] dataset-like export: {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
