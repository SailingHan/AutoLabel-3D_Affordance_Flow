#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, env=None):
    print("[cmd]", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    root_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        "Run masked-role TraceForge inference on a single trajectory with current recommended defaults"
    )
    ap.add_argument("--traj-dir", type=Path, required=True)
    ap.add_argument("--traceforge-root", type=Path, default=Path("/home/zhy/data/TraceForge"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/zhy/data/outputs_run_pipeline/traceforge_formal"),
    )
    ap.add_argument("--checkpoint", type=Path, default=Path("/home/zhy/data/TraceForge/checkpoints/tapip3d_final.pth"))
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--max-num-frames", type=int, default=2000)
    ap.add_argument("--future-len", type=int, default=128)
    ap.add_argument("--frame-drop-rate", type=int, default=4)
    ap.add_argument("--query-mode", type=str, default="masked_roles", choices=["grid", "masked_roles"])
    ap.add_argument("--use-known-depth", action="store_true")
    ap.add_argument("--use-known-intrinsics", action="store_true")
    ap.add_argument("--auto-raise-future-len-to-seq-len", action="store_true")
    args, extra = ap.parse_known_args()

    infer_script = root_dir / "code" / "traceforge" / "infer.py"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    infer_cmd = [
        sys.executable,
        str(infer_script),
        "--video_path",
        str(args.traj_dir),
        "--out_dir",
        str(args.out_dir),
        "--input_layout",
        "rvideo_traj_dataset",
        "--query_mode",
        args.query_mode,
        "--checkpoint",
        str(args.checkpoint),
        "--fps",
        str(args.fps),
        "--max_num_frames",
        str(args.max_num_frames),
        "--future_len",
        str(args.future_len),
        "--frame_drop_rate",
        str(args.frame_drop_rate),
    ]
    if args.use_known_depth:
        infer_cmd.append("--use_known_depth")
    if args.use_known_intrinsics:
        infer_cmd.append("--use_known_intrinsics")
    if args.auto_raise_future_len_to_seq_len:
        infer_cmd.append("--auto_raise_future_len_to_seq_len")
    infer_cmd.extend(extra)
    env = os.environ.copy()
    env["TRACEFORGE_ROOT"] = str(args.traceforge_root.resolve())
    run_cmd(infer_cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
