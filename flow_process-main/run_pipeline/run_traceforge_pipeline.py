#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


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
        raise SystemExit(
            f"missing LLM config fields for provider={provider}: {', '.join(missing)}. "
            "Pass --model/--base-url/--api-key-env explicitly or use a provider preset with defaults."
        )
    return {k: str(v) for k, v in resolved.items()}


def resolve_model_seq_len(traceforge_root: Path, checkpoint: Path | None = None) -> int:
    sys.path.insert(0, str(traceforge_root))
    import models  # type: ignore

    ckpt = checkpoint or (traceforge_root / "checkpoints" / "tapip3d_final.pth")
    model, _ = models.from_pretrained(ckpt)
    return int(getattr(model, "seq_len", 0) or 0)


def run_cmd(cmd, env=None):
    print("[cmd]", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.run(cmd, check=True, env=env)


def resolve_exported_task_filter(dataset_root: Path, source_task_filter: str) -> str:
    requested = {x.strip() for x in str(source_task_filter).split(",") if x.strip()}
    if not requested:
        return ""

    matched_action_dirs: set[str] = set()
    for action_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if action_dir.name.startswith("_"):
            continue
        for traj_dir in sorted(p for p in action_dir.iterdir() if p.is_dir() and p.name.startswith("traj_")):
            llm_path = traj_dir / "llm_semantics.json"
            if not llm_path.exists():
                continue
            try:
                with open(llm_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            if str(payload.get("source_task", "")).strip() in requested:
                matched_action_dirs.add(action_dir.name)
                break
    return ",".join(sorted(matched_action_dirs))


def main():
    root_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        "Run the current masked-role pipeline with TraceForge as a dependency"
    )
    ap.add_argument("--stage", choices=["all", "llm", "adapt", "infer"], default="all")
    ap.add_argument("--source-mode", choices=["rvideo", "bridge_raw"], default="rvideo")
    ap.add_argument("--traceforge-root", type=Path, default=Path("/home/zhy/data/TraceForge"))
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/zhy/data/TraceForge/checkpoints/tapip3d_final.pth"),
    )
    ap.add_argument("--video-root", type=Path, default=Path("/home/zhy/data/datasets/video"))
    ap.add_argument("--bridge-root", type=Path, default=Path("/home/zhy/data/datasets/bridge_raw"))
    ap.add_argument("--dataset-root", type=Path, default=Path("/home/zhy/data/datasets"))
    ap.add_argument(
        "--traceforge-output",
        type=Path,
        default=Path("/home/zhy/data/outputs_run_pipeline/traceforge_auto"),
    )
    ap.add_argument("--task-filter", type=str, default="")
    ap.add_argument("--frames-per-video", type=int, default=6)
    ap.add_argument("--max-samples-per-task", type=int, default=0)
    ap.add_argument("--llm-provider", choices=sorted(LLM_PROVIDER_PRESETS), default="moonshot")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--base-url", type=str, default=None)
    ap.add_argument("--api-key-env", type=str, default=None)
    ap.add_argument("--semantic-mode", choices=["rules", "rules_then_llm", "llm"], default="llm")
    ap.add_argument("--semantic-timeout-sec", type=int, default=120)
    ap.add_argument("--semantic-retries", type=int, default=2)
    ap.add_argument("--default-tool", type=str, default="gripper")
    ap.add_argument("--default-target", type=str, default="object")
    ap.add_argument("--query-mode", type=str, default="masked_roles", choices=["grid", "masked_roles"])
    ap.add_argument("--frame-drop-rate", type=int, default=4)
    ap.add_argument("--future-len", type=int, default=128)
    ap.add_argument("--auto-raise-future-len-to-seq-len", action="store_true")
    ap.add_argument("--max-frames-per-video", type=int, default=50)
    ap.add_argument("--max-num-frames", type=int, default=2000)
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--overwrite-llm", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--use-known-depth", action="store_true")
    ap.add_argument("--use-known-intrinsics", action="store_true")
    args, extra = ap.parse_known_args()
    llm_config = resolve_llm_config(args.llm_provider, args.model, args.base_url, args.api_key_env)

    traceforge_root = args.traceforge_root.resolve()
    llm_script = root_dir / "code" / "llm_to_dataset.py"
    bridge_adapter_script = root_dir / "code" / "adapters" / "bridge_data_adapter.py"
    infer_script = root_dir / "code" / "traceforge" / "infer.py"

    env = os.environ.copy()
    env["TRACEFORGE_ROOT"] = str(traceforge_root)
    python_bin = sys.executable

    if args.source_mode == "bridge_raw" and args.stage in {"all", "adapt"}:
        adapt_cmd = [
            python_bin,
            str(bridge_adapter_script),
            "--bridge-root",
            str(args.bridge_root),
            "--output-root",
            str(args.dataset_root),
            "--default-tool",
            args.default_tool,
            "--default-target",
            args.default_target,
            "--semantic-mode",
            args.semantic_mode,
            "--semantic-provider",
            llm_config["provider"],
            "--semantic-model",
            llm_config["model"],
            "--semantic-base-url",
            llm_config["base_url"],
            "--semantic-api-key-env",
            llm_config["api_key_env"],
            "--semantic-timeout-sec",
            str(args.semantic_timeout_sec),
            "--semantic-retries",
            str(args.semantic_retries),
            "--max-episodes",
            str(args.max_samples_per_task),
            "--max-frames",
            str(args.max_frames_per_video),
        ]
        if args.task_filter:
            adapt_cmd.extend(["--task-label", args.task_filter.split(",")[0].strip()])
        if args.overwrite_llm:
            adapt_cmd.append("--overwrite")
        run_cmd(adapt_cmd, env=env)

    if args.source_mode == "rvideo" and args.stage in {"all", "llm"}:
        llm_cmd = [
            python_bin,
            str(llm_script),
            "--video_root",
            str(args.video_root),
            "--output_dir",
            str(args.dataset_root),
            "--model",
            llm_config["model"],
            "--base_url",
            llm_config["base_url"],
            "--frames_per_video",
            str(args.frames_per_video),
            "--max_samples_per_task",
            str(args.max_samples_per_task),
            "--api_key_env",
            llm_config["api_key_env"],
        ]
        if args.task_filter:
            llm_cmd.extend(["--task_filter", args.task_filter])
        if args.overwrite_llm:
            llm_cmd.append("--overwrite")
        run_cmd(llm_cmd, env=env)

    if args.stage in {"all", "infer"}:
        infer_task_filter = args.task_filter
        if args.source_mode == "rvideo" and args.stage == "all" and args.task_filter:
            infer_task_filter = resolve_exported_task_filter(args.dataset_root.resolve(), args.task_filter)
            print(f"[info] exported infer task_filter = {infer_task_filter or '<none>'}")

        model_seq_len = resolve_model_seq_len(traceforge_root, args.checkpoint.resolve())
        effective_future_len = int(args.future_len)
        if effective_future_len < model_seq_len:
            if args.auto_raise_future_len_to_seq_len:
                print(
                    f"[warn] future_len={effective_future_len} < model.seq_len={model_seq_len}; "
                    f"raising effective future_len to {model_seq_len}"
                )
                effective_future_len = model_seq_len
            else:
                raise SystemExit(
                    f"future_len={effective_future_len} is smaller than model.seq_len={model_seq_len}. "
                    "This would produce static coords because the tracker would not enter any tracking window. "
                    "Set --future-len >= model.seq_len or pass --auto-raise-future-len-to-seq-len."
                )

        args.traceforge_output.mkdir(parents=True, exist_ok=True)
        infer_cmd = [
            python_bin,
            str(infer_script),
            "--video_path",
            str(args.dataset_root),
            "--out_dir",
            str(args.traceforge_output),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--batch_process",
            "--input_layout",
            "rvideo_traj_dataset",
            "--query_mode",
            args.query_mode,
            "--frame_drop_rate",
            str(args.frame_drop_rate),
            "--future_len",
            str(effective_future_len),
            "--max_frames_per_video",
            str(args.max_frames_per_video),
            "--max_num_frames",
            str(args.max_num_frames),
            "--fps",
            str(args.fps),
        ]
        if args.auto_raise_future_len_to_seq_len:
            infer_cmd.append("--auto_raise_future_len_to_seq_len")
        if infer_task_filter:
            infer_cmd.extend(["--task_filter", infer_task_filter])
        if args.skip_existing:
            infer_cmd.append("--skip_existing")
        if args.use_known_depth:
            infer_cmd.append("--use_known_depth")
        if args.use_known_intrinsics:
            infer_cmd.append("--use_known_intrinsics")
        infer_cmd.extend(extra)
        run_cmd(infer_cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
