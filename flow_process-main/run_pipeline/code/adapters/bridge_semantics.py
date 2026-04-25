#!/usr/bin/env python3
"""BridgeData episode semantics: rule candidates plus optional LLM normalization."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
from PIL import Image


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


def resolve_llm_config(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, str]:
    if provider not in LLM_PROVIDER_PRESETS:
        raise ValueError(f"unknown LLM provider: {provider}")
    preset = LLM_PROVIDER_PRESETS[provider]
    resolved = {
        "provider": provider,
        "model": model or preset.get("model"),
        "base_url": base_url or preset.get("base_url"),
        "api_key_env": api_key_env or preset.get("api_key_env"),
    }
    missing = [k for k in ("model", "base_url", "api_key_env") if not resolved.get(k)]
    if missing:
        raise ValueError(
            f"missing LLM config fields for provider={provider}: {', '.join(missing)}. "
            "Pass --model/--base-url/--api-key-env explicitly or use a preset provider."
        )
    return {k: str(v) for k, v in resolved.items()}


OBJECT_ALIASES = {
    "utensils": "utensil",
    "silverware": "utensil",
    "spoon": "spoon",
    "fork": "fork",
    "spatula": "spatula",
    "cup": "cup",
    "mug": "cup",
    "pot": "pot",
    "pan": "pan",
    "plate": "plate",
    "bowl": "bowl",
    "drawer": "drawer",
    "door": "door",
    "cloth": "cloth",
    "block": "block",
}

ACTION_ALIASES = {
    "pnp": "pick_up",
    "pick": "pick_up",
    "place": "place",
    "put": "place",
    "push": "push",
    "pull": "pull",
    "open": "open",
    "close": "close",
    "sweep": "sweep",
    "stack": "stack",
    "fold": "fold",
    "flip": "flip",
}


def normalize_label(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_+ -]+", "", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unknown"


def sample_image_paths(episode_dir: Path, max_frames: int = 4) -> list[Path]:
    files = sorted(episode_dir.glob("rgb_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        image_dirs = [p for p in episode_dir.rglob("*") if p.is_dir() and p.name.lower() in {"images0", "images"}]
        for image_dir in image_dirs[:1]:
            files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not files:
        return []
    if len(files) <= max_frames:
        return files
    picks = sorted({round(i * (len(files) - 1) / (max_frames - 1)) for i in range(max_frames)})
    return [files[int(i)] for i in picks]


def image_data_url(path: Path, quality: int = 80) -> str:
    image = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def infer_rule_candidates(task_name: str, language: str, episode_dir: Path, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    text = normalize_label(" ".join(x for x in [task_name, language, str(episode_dir)] if x))
    tokens = [x for x in re.split(r"[_/\\s-]+", text) if x]
    phrases = [text.replace("_", " "), text]

    action = "interact"
    for token in tokens:
        if token in ACTION_ALIASES:
            action = ACTION_ALIASES[token]
            break

    target = ""
    for alias, canonical in OBJECT_ALIASES.items():
        if alias in tokens or any(re.search(rf"\b{re.escape(alias)}\b", phrase) for phrase in phrases):
            target = canonical
            break
    if not target:
        for token in tokens:
            if token not in ACTION_ALIASES and not token.isdigit() and token not in {"raw", "traj", "group", "scripted"}:
                target = token
                break
    target = target or "object"

    robot_state = False
    if meta:
        state_shape = meta.get("state_shape")
        action_shape = meta.get("action_shape")
        robot_state = bool(state_shape or action_shape)
    tool = "gripper" if robot_state or "scripted_raw" in str(episode_dir) else "hand"

    folder_label = normalize_label(task_name or f"{action}_{target}")
    return {
        "core_description": f"{action.replace('_', ' ')} {target}",
        "action": action,
        "tool": tool,
        "target": target,
        "interaction": f"{tool} {action.replace('_', ' ')} {target}",
        "folder_label": folder_label,
        "_rule_context": {
            "task_name": task_name,
            "language": language,
            "tokens": tokens[:40],
            "episode_dir": str(episode_dir),
        },
    }


def build_llm_payload(model: str, candidates: dict[str, Any], frame_paths: list[Path]) -> dict[str, Any]:
    contract = {
        "core_description": "short visible manipulation description",
        "action": "short verb phrase, snake_case if multiword",
        "tool": "active manipulator or tool, e.g. gripper, hand, spatula",
        "target": "primary manipulated object, concrete visual noun",
        "interaction": "short phrase describing tool-target interaction",
        "folder_label": "compact action/object label",
    }
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are standardizing semantics for a BridgeData robot episode used for "
                "GroundingDINO+SAM role masks. Prefer concrete visible nouns over broad classes. "
                "For robot scripted data, the active manipulator is usually a gripper unless a handheld tool is visible. "
                "Return strict JSON only.\n\n"
                f"Rule candidates:\n{json.dumps(candidates, indent=2)}\n\n"
                f"Required JSON shape:\n{json.dumps(contract, indent=2)}"
            ),
        }
    ]
    for path in frame_paths:
        content.append({"type": "text", "text": f"sample_frame={path.name}"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
    return {"model": model, "messages": [{"role": "user", "content": content}]}


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_llm_standardizer(
    *,
    candidates: dict[str, Any],
    frame_paths: list[Path],
    provider: str,
    model: str,
    base_url: str,
    api_key_env: str,
    timeout_sec: int,
    retries: int,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env {api_key_env}")
    payload = build_llm_payload(model, candidates, frame_paths)
    print(
        f"[bridge_semantics] calling LLM provider={provider} model={model} frames={len(frame_paths)} "
        f"candidate={candidates.get('tool')}->{candidates.get('target')}",
        flush=True,
    )
    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            print(f"[bridge_semantics] LLM attempt {attempt}/{max(1, retries)}", flush=True)
            resp = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout_sec,
            )
            break
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            print(f"[bridge_semantics] LLM timeout attempt {attempt}: {exc}", flush=True)
    else:
        raise last_exc if last_exc is not None else RuntimeError("LLM request failed")
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    out = extract_json_object(text)
    out["_raw_response_id"] = data.get("id")
    out["_model"] = data.get("model")
    print(
        f"[bridge_semantics] LLM result tool={out.get('tool')} target={out.get('target')} "
        f"action={out.get('action')}",
        flush=True,
    )
    return out


def sanitize_semantics(payload: dict[str, Any], fallback: dict[str, Any]) -> dict[str, str]:
    out = {}
    for key in ("core_description", "action", "tool", "target", "interaction", "folder_label"):
        value = str(payload.get(key) or fallback.get(key) or "").strip()
        if key in {"action", "tool", "target", "folder_label"}:
            value = normalize_label(value)
        out[key] = value or str(fallback.get(key, "unknown"))
    return out


def generate_semantics(
    *,
    episode_dir: Path,
    task_name: str,
    language: str,
    meta: dict[str, Any] | None,
    mode: str,
    provider: str,
    model: str,
    base_url: str,
    api_key_env: str,
    timeout_sec: int = 60,
    retries: int = 2,
) -> tuple[dict[str, str], dict[str, Any]]:
    candidates = infer_rule_candidates(task_name, language, episode_dir, meta=meta)
    frame_paths = sample_image_paths(episode_dir)
    debug = {
        "mode": mode,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "rule_candidates": candidates,
        "sample_frames": [str(p) for p in frame_paths],
        "llm_used": False,
    }

    final_payload: dict[str, Any] = candidates
    if mode in {"llm", "rules_then_llm"}:
        try:
            final_payload = call_llm_standardizer(
                candidates=candidates,
                frame_paths=frame_paths,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
                timeout_sec=timeout_sec,
                retries=retries,
            )
            debug["llm_used"] = True
            debug["llm_payload"] = {k: v for k, v in final_payload.items() if not k.startswith("_")}
        except Exception as exc:
            if mode == "llm":
                raise
            debug["llm_error"] = str(exc)
            final_payload = candidates

    return sanitize_semantics(final_payload, candidates), debug


def main() -> int:
    parser = argparse.ArgumentParser("Generate BridgeData llm_semantics.json for one adapted episode")
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--task-name", type=str, default="")
    parser.add_argument("--language", type=str, default="")
    parser.add_argument("--mode", choices=["rules", "rules_then_llm", "llm"], default="rules_then_llm")
    parser.add_argument("--provider", choices=sorted(LLM_PROVIDER_PRESETS), default="moonshot")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    llm_config = resolve_llm_config(args.provider, args.model, args.base_url, args.api_key_env)
    meta_path = args.episode_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    language = args.language or ((args.episode_dir / "language.txt").read_text().strip() if (args.episode_dir / "language.txt").exists() else "")
    semantics, debug = generate_semantics(
        episode_dir=args.episode_dir,
        task_name=args.task_name or args.episode_dir.parent.name,
        language=language,
        meta=meta,
        mode=args.mode,
        provider=llm_config["provider"],
        model=llm_config["model"],
        base_url=llm_config["base_url"],
        api_key_env=llm_config["api_key_env"],
        timeout_sec=args.timeout_sec,
        retries=args.retries,
    )
    print(json.dumps({"llm_result": semantics, "debug": debug}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
