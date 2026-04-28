<div align="center">

# Masked-Role TraceForge

**Semantic role-aware 2D/3D trajectory generation for robot and human manipulation videos.**

<p>
  <a href="#quick-start"><img alt="Quick Start" src="https://img.shields.io/badge/quick%20start-ready-1f6feb?style=flat&labelColor=0d1117" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-3776AB?style=flat&logo=python&logoColor=white&labelColor=0d1117" />
  <img alt="CUDA" src="https://img.shields.io/badge/cuda-required-76B900?style=flat&logo=nvidia&logoColor=white&labelColor=0d1117" />
  <a href="#license"><img alt="License" src="https://img.shields.io/badge/license-MIT-f85149?style=flat&labelColor=0d1117" /></a>
</p>

<p>
  <img alt="TraceForge" src="https://img.shields.io/badge/TraceForge-tracking-8b949e?style=flat&labelColor=0d1117" />
  <img alt="GroundingDINO" src="https://img.shields.io/badge/GroundingDINO-detection-f97316?style=flat&labelColor=0d1117" />
  <img alt="SAM2" src="https://img.shields.io/badge/SAM2-segmentation-22c55e?style=flat&labelColor=0d1117" />
  <img alt="LLM" src="https://img.shields.io/badge/LLM-multi--provider-a855f7?style=flat&labelColor=0d1117" />
  <img alt="Teacher Targets" src="https://img.shields.io/badge/output-2D%2F3D%20teacher%20targets-06b6d4?style=flat&labelColor=0d1117" />
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#highlights">Highlights</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#bridgedata-branch">BridgeData</a> ·
  <a href="#validation">Validation</a> ·
  <a href="#citation">Citation</a> ·
  <a href="#license">License</a>
</p>

</div>

---

## Overview

`Masked-Role TraceForge` is a lightweight workflow layer built around **TraceForge**, **GroundingDINO**, **SAM2**, and an OpenAI-compatible multimodal LLM API.

It converts raw robot or human manipulation videos into **role-aware 2D/3D motion traces** that can be used as self-supervised teacher targets for:

- trajectory prediction,
- TraceGen-style trace learning,
- embodied world-model pretraining,
- robot manipulation representation learning,
- semantic motion understanding.

The core design principle is:

> Keep the official TraceForge tracking backbone intact, and replace only the query source with task-relevant, mask-constrained role sampling.

Instead of sampling query points from the entire image, this workflow first identifies semantic interaction roles such as `hand`, `tool`, and `target`, generates foreground masks for these roles, and then tracks only task-relevant points through TraceForge.

---

## Highlights

<table>
  <tr>
    <td><b>Semantic role parsing</b></td>
    <td>Uses an OpenAI-compatible LLM API to infer action, tool, target, and interaction semantics from task context and sampled frames.</td>
  </tr>
  <tr>
    <td><b>Mask-constrained query sampling</b></td>
    <td>Uses GroundingDINO + SAM2 to generate role masks and sample foreground-only query points.</td>
  </tr>
  <tr>
    <td><b>TraceForge-compatible tracking</b></td>
    <td>Preserves the original TraceForge depth, pose, and <code>PointTracker3D</code> tracking path.</td>
  </tr>
  <tr>
    <td><b>Role-preserving flow export</b></td>
    <td>Exports <code>hand_flow.npy</code>, <code>tool_flow.npy</code>, <code>target_flow.npy</code>, and role-indexed query metadata.</td>
  </tr>
  <tr>
    <td><b>BridgeData support</b></td>
    <td>Includes a raw BridgeData adapter for self-supervised teacher-target generation from robot videos.</td>
  </tr>
  <tr>
    <td><b>Multi-provider LLM interface</b></td>
    <td>Supports Moonshot/Kimi, OpenAI, and custom OpenAI-compatible endpoints through a unified provider interface.</td>
  </tr>
</table>

---

## How It Works

The workflow is organized around a conservative modification of TraceForge:

1. **Standardize videos into local episodes.**  
   Raw robot or human videos are converted into an episode layout containing RGB frames, task context, optional actions/states, and optional camera metadata.

2. **Parse task semantics.**  
   A multimodal LLM extracts task-level semantics such as `action`, `tool`, `target`, and `interaction`.

3. **Generate role masks.**  
   GroundingDINO proposes role-conditioned boxes and SAM2 converts those boxes into foreground masks.

4. **Sample task-relevant query points.**  
   Query points are sampled only from the role masks rather than from the full image grid.

5. **Track with TraceForge.**  
   The official TraceForge tracking path is preserved, including depth/pose processing and `PointTracker3D`.

6. **Export role-aware teacher targets.**  
   The final outputs preserve role identity through files such as `hand_flow.npy`, `tool_flow.npy`, `target_flow.npy`, and `teacher_targets.npz`.

---

## Design Philosophy

This workflow is intentionally conservative. It does **not** fork or rewrite the TraceForge tracker.

| Design choice | Motivation |
|---|---|
| Keep TraceForge as an external dependency | Avoid copying or modifying the full tracking stack. |
| Replace only the 2D query source | Make the workflow semantic-aware while preserving official tracking behavior. |
| Fail fast when masks are missing by default | Avoid silently exporting low-quality full-image grid traces as role-aware data. |
| Preserve role identity in outputs | Let downstream models distinguish hand/tool/target motion explicitly. |
| Keep BridgeData outputs separate | Prevent probe data, TraceForge outputs, and teacher assets from polluting the mainline. |

---

## Repository Layout

```text
<PATH_TO_WORKSPACE>/
├── run_book/
│   └── README.md                         # Global runbook and workflow reference
│
├── run_pipeline/
│   ├── README.md                         # Current workflow bundle documentation
│   ├── run_traceforge_pipeline.py         # Unified entrypoint
│   ├── run_single_traj.py                 # Single-trajectory runner
│   └── code/
│       ├── adapters/
│       │   └── bridge_data_adapter.py     # BridgeData raw-video adapter
│       ├── llm_to_dataset.py              # LLM semantic labeling and trajectory export
│       ├── dino_hoi_detector.py           # GroundingDINO + SAM2 role-mask generation
│       ├── tools/
│       │   ├── export_teacher_targets.py
│       │   └── filter_teacher_targets.py
│       └── traceforge/
│           ├── infer.py                   # Masked-role TraceForge inference wrapper
│           ├── visualize_sample_2d_flow.py
│           ├── visualize_single_image.py
│           └── utils/
│
├── datasets/                              # Local standardized episodes
├── outputs_run_pipeline/                  # Global output root
├── TraceForge/                            # External dependency repo
├── sam2/                                  # External dependency repo
└── GroundingDINO/                         # External dependency repo
```

> [!IMPORTANT]
> `run_pipeline/` is only the workflow delta. It is not a full dependency closure. TraceForge, SAM2, and GroundingDINO should stay as external repositories.

---

## Quick Start

### 1. Clone dependencies

```bash
git clone https://github.com/Yoonkyo/TraceForge.git <PATH_TO_WORKSPACE>/TraceForge
git clone https://github.com/facebookresearch/sam2.git <PATH_TO_WORKSPACE>/sam2
git clone https://github.com/idea-research/groundingdino.git <PATH_TO_WORKSPACE>/GroundingDINO
```

### 2. Create the TraceForge environment

```bash
cd <PATH_TO_WORKSPACE>/TraceForge
conda create -n traceforge python=3.11
conda activate traceforge
bash setup_env.sh
```

### 3. Install SAM2 and GroundingDINO

```bash
cd <PATH_TO_WORKSPACE>/sam2
pip install -e .

cd <PATH_TO_WORKSPACE>/GroundingDINO
pip install -e .
```

Install additional utilities if they are not already available:

```bash
pip install requests pillow opencv-python matplotlib loguru rich tqdm mediapy viser
```

> [!NOTE]
> `run_pipeline/` itself is not a Python package. Do not run `pip install -e .` inside `run_pipeline/`. Run its scripts directly from the activated `traceforge` environment.

### 4. Prepare checkpoints

Required checkpoint paths:

```text
<PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth
<PATH_TO_WORKSPACE>/sam2/checkpoints/sam2.1_hiera_large.pt
<PATH_TO_WORKSPACE>/GroundingDINO/checkpoint/groundingdino_swint_ogc.pth
```

Example downloads:

```bash
mkdir -p <PATH_TO_WORKSPACE>/TraceForge/checkpoints
wget -O <PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth \
  https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth

mkdir -p <PATH_TO_WORKSPACE>/GroundingDINO/checkpoint
wget -O <PATH_TO_WORKSPACE>/GroundingDINO/checkpoint/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

For SAM2.1 Hiera-L, follow the official SAM2 checkpoint instructions and place the file at:

```text
<PATH_TO_WORKSPACE>/sam2/checkpoints/sam2.1_hiera_large.pt
```

### 5. Check CUDA and imports

```bash
which nvcc
echo $CUDA_HOME
```

If `CUDA_HOME` is empty and `nvcc` is under `/usr/local/cuda/bin/nvcc`, use:

```bash
export CUDA_HOME=/usr/local/cuda
```

Then verify imports:

```bash
python - <<'PY'
import torch
import sam2
import groundingdino
import requests, PIL, cv2, matplotlib, viser

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("imports ok")
PY
```

---

## LLM Semantics API

The LLM is used only during semantic labeling and adaptation. TraceForge inference does **not** call the LLM; it only reads the generated `llm_semantics.json`.

The API client follows the OpenAI-compatible `/chat/completions` protocol. Provider presets define endpoint and API-key defaults, while the concrete model can always be overridden with `--model`.

### Provider presets

| Provider | Flag | Default base URL | API key env | Model behavior |
|---|---|---|---|---|
| Moonshot / Kimi | `--llm-provider moonshot` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` | Default: `kimi-k2.5` |
| OpenAI | `--llm-provider openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | Pass with `--model` |
| Custom | `--llm-provider custom` | User-defined | User-defined | Pass all fields explicitly |

Set API keys in the active `traceforge` shell:

```bash
export MOONSHOT_API_KEY=<YOUR_MOONSHOT_API_KEY>
# or
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

Unified provider rule:

```text
--llm-provider   selects the provider preset
--model          selects the concrete model name
--base-url       overrides endpoint, mainly for custom providers
--api-key-env    overrides key variable, mainly for custom providers
```

---

## Semantics Source

`code/traceforge/infer.py` never calls the LLM during inference.

Semantics are loaded in this priority order:

1. Read `<traj_dir>/llm_semantics.json`.
2. If missing, parse the parent task folder name as a fallback.

The intended automated flow is:

```text
llm_to_dataset.py
  └── writes llm_semantics.json
        └── consumed by masked-role TraceForge inference
```

---

## Mask Source

For `masked_roles`, masks are generated online:

1. GroundingDINO predicts role boxes from text prompts.
2. SAM2 segments masks from predicted boxes.
3. Query points are sampled only from foreground pixels inside the role masks.

By default, there is no silent fallback to full-image grid queries when role masks fail. This keeps the exported traces semantically strict and easier to audit.

---

## Tracker Constraints

Current tracker configuration:

```text
checkpoint: <PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth
model:      PointTracker3D
seq_len:    16
```

Required condition:

```text
future_len >= model.seq_len
```

If `future_len < seq_len`, the tracker will not enter a valid tracking window and may produce static coordinates.

Recommended defaults:

```text
--input_layout rvideo_traj_dataset
--query_mode masked_roles
--fps 1
--max_num_frames 2000
--future_len 128
--frame_drop_rate 4
```

---

## Usage

### Full automated workflow

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py \
  --stage all \
  --video-root <PATH_TO_WORKSPACE>/datasets/video \
  --dataset-root <PATH_TO_WORKSPACE>/datasets \
  --traceforge-output <PATH_TO_WORKSPACE>/outputs_run_pipeline/traceforge_auto \
  --checkpoint <PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth \
  --query-mode masked_roles \
  --fps 1 \
  --future-len 128 \
  --frame-drop-rate 4
```

### Inference only on existing trajectories

Use this when `llm_semantics.json` already exists.

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py \
  --stage infer \
  --dataset-root <PATH_TO_WORKSPACE>/datasets \
  --traceforge-output <PATH_TO_WORKSPACE>/outputs_run_pipeline/traceforge_auto \
  --checkpoint <PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth \
  --task-filter close_oven \
  --query-mode masked_roles
```

### Single trajectory

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_single_traj.py \
  --traj-dir <PATH_TO_WORKSPACE>/datasets/close_oven/traj_001 \
  --out-dir <PATH_TO_WORKSPACE>/outputs_run_pipeline/traceforge_formal \
  --checkpoint <PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth
```

---

## BridgeData Branch

The BridgeData branch is for **self-supervised data generation**, not policy learning.

BridgeData provides raw robot videos and task distributions. This workflow adapts those videos into local episodes, generates semantic role annotations, runs masked-role TraceForge tracking, and exports teacher targets for later TraceGen, world-model, or trace-prediction pretraining.

Action and state arrays are preserved by the adapter for later weak supervision or downstream fine-tuning, but this branch does not connect an execution layer.

### Episode format

```text
bridge_episodes/<task_label>/traj_000000/
├── rgb_0.png
├── rgb_1.png
├── meta.json
├── action.npy
├── state.npy
├── language.txt
├── llm_semantics.json
└── camera_in.npy        # optional, only if available in the source
```

### Recommended BridgeData entrypoint

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py \
  --source-mode bridge_raw \
  --stage adapt \
  --bridge-root <PATH_TO_BRIDGEDATA_RAW> \
  --dataset-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes \
  --task-filter pnp_utensils \
  --max-samples-per-task 5 \
  --max-frames-per-video 16 \
  --llm-provider moonshot \
  --semantic-mode llm \
  --model kimi-k2.5 \
  --semantic-timeout-sec 180 \
  --semantic-retries 3 \
  --overwrite-llm
```

OpenAI-compatible example:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py \
  --source-mode bridge_raw \
  --stage adapt \
  --bridge-root <PATH_TO_BRIDGEDATA_RAW> \
  --dataset-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes \
  --task-filter pnp_utensils \
  --max-samples-per-task 5 \
  --max-frames-per-video 16 \
  --llm-provider openai \
  --model <OPENAI_VISION_MODEL> \
  --semantic-mode llm \
  --overwrite-llm
```

Custom provider example:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py \
  --source-mode bridge_raw \
  --stage adapt \
  --bridge-root <PATH_TO_BRIDGEDATA_RAW> \
  --dataset-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes \
  --task-filter pnp_utensils \
  --max-samples-per-task 5 \
  --max-frames-per-video 16 \
  --llm-provider custom \
  --base-url <YOUR_BASE_URL> \
  --api-key-env <YOUR_KEY_ENV> \
  --model <YOUR_MODEL_NAME> \
  --semantic-mode llm \
  --overwrite-llm
```

### Semantic modes

| Mode | Behavior |
|---|---|
| `--semantic-mode llm` | Require the LLM call to succeed. Fail if the API times out or errors. |
| `--semantic-mode rules_then_llm` | Try the LLM first, then fall back to rule candidates if the API fails. |
| `--semantic-mode rules` | Do not call the API. Use deterministic rule extraction only. |

For BridgeData, the LLM request includes task metadata plus sampled full RGB frames. The default path is one episode per request, not one giant batch request.

### Stage semantics

| Stage | What it does | What it does not do |
|---|---|---|
| `--stage adapt` | Scans raw BridgeData, exports local episodes, writes `llm_semantics.json` | Does not run TraceForge inference |
| `--stage infer` | Runs masked-role TraceForge on existing adapted episodes | Does not regenerate semantics |
| `--stage all` | Runs `adapt` then `infer` | Use only when you intentionally want to regenerate semantics |

After changing or regenerating `llm_semantics.json`, rerun `--stage infer`; existing TraceForge output directories are not automatically updated.

### BridgeData inference

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py \
  --source-mode bridge_raw \
  --stage infer \
  --dataset-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes \
  --traceforge-output <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_traceforge \
  --query-mode masked_roles \
  --fps 1 \
  --future-len 128 \
  --frame-drop-rate 4 \
  --max-num-frames 2000
```

### Optional grid fallback for probe runs

By default, missing role masks fail fast. For exploratory BridgeData probes where keeping a sample is more important than strict role-only queries, enable per-frame grid fallback:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py \
  --source-mode bridge_raw \
  --stage infer \
  --dataset-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes \
  --traceforge-output <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_traceforge \
  --query-mode masked_roles \
  --masked_role_fallback grid
```

Fallback metadata is recorded per query frame:

```text
role_queries/<episode>_<frame>_query_source.json
```

Possible values:

```text
masked_roles
fallback_grid
```

Grid-fallback queries use role id `0` (`generic`). They will not appear in `hand_flow.npy`, `tool_flow.npy`, or `target_flow.npy`; inspect `selected_traces.npy` and the query source JSON for those frames.

### Default BridgeData outputs

```text
<PATH_TO_WORKSPACE>/outputs_run_pipeline/
├── bridge_episodes/          # adapted local episodes
├── bridge_traceforge/        # raw TraceForge outputs
└── bridge_teacher_targets/   # training assets
```

Each teacher-target episode directory contains:

```text
teacher_targets.npz
selected_traces.npy
hand_flow.npy
tool_flow.npy
target_flow.npy
query_role_id.npy
query_role_name.json
```

`teacher_targets.npz` contains at least:

```text
trace_3d
trace_2d
valid_steps
visibility
query_frame_index
role_id
```

---

## Verify LLM Usage

After `--stage adapt` or `--stage llm`, inspect one generated semantics file:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("<PATH_TO_TRAJ>/llm_semantics.json")
data = json.loads(path.read_text())

print(json.dumps(data.get("llm_result", data), indent=2, ensure_ascii=False))
print(json.dumps(data.get("semantic_debug", {}), indent=2, ensure_ascii=False))
PY
```

For BridgeData outputs, `semantic_debug.llm_used` should be `true` when the LLM API was actually used. The file should contain concrete `tool` and `target` values, such as `gripper` and `spatula`, rather than only generic defaults like `object`.

---

## Validation

Use visual inspection and trace consistency checks as the primary success criteria.

Recommended validation order:

1. Inspect `llm_semantics.json` and confirm `semantic_debug.llm_used=true`.
2. Inspect adapter preview images and semantic labels.
3. Inspect role masks and query overlays.
4. Inspect `selected_traces.npy`, `hand_flow.npy`, `tool_flow.npy`, and `target_flow.npy`.
5. Run TraceForge 2D and 3D result checkers.
6. Use the single-image 3D Viser viewer on representative samples.

> [!CAUTION]
> Do not use the official single-sample viewer as the only success criterion for `masked_roles`. Role-aware traces should be checked with both 2D overlays and 3D consistency tools.

### TraceForge checkers

```bash
python <PATH_TO_WORKSPACE>/TraceForge/checker/batch_process_result_checker.py \
  <PATH_TO_WORKSPACE>/outputs_run_pipeline/traceforge_auto

python <PATH_TO_WORKSPACE>/TraceForge/checker/batch_process_result_checker_3d.py \
  <PATH_TO_WORKSPACE>/outputs_run_pipeline/traceforge_auto
```

### Batch 2D visualization from BridgeData outputs

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/code/traceforge/visualize_sample_2d_flow.py \
  --batch-output-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_traceforge \
  --adapter-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes \
  --vis-output-dir <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_visualizations
```

### Single 2D overlay

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/code/traceforge/visualize_sample_2d_flow.py \
  --sample-npz <PATH_TO_EPISODE_DIR>/samples/<EPISODE_NAME>_<QUERY_FRAME>.npz \
  --episode-dir <PATH_TO_EPISODE_DIR> \
  --output <PATH_TO_EPISODE_DIR>/vis_2d/<EPISODE_NAME>_<QUERY_FRAME>.mp4
```

### Single-image 3D viewer

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/code/traceforge/visualize_single_image.py \
  --npz_path <PATH_TO_EPISODE_DIR>/samples/<EPISODE_NAME>_<QUERY_FRAME>.npz \
  --image_path <PATH_TO_EPISODE_DIR>/images/<EPISODE_NAME>_<QUERY_FRAME>.png \
  --depth_path <PATH_TO_EPISODE_DIR>/depth/<EPISODE_NAME>_<QUERY_FRAME>.png \
  --port 8080
```

---

## Quality Filtering

Quality filtering is optional at the current probe stage. It is intended for later large-scale teacher-target generation.

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/code/tools/filter_teacher_targets.py \
  --teacher-output <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_teacher_targets
```

The filter writes:

```text
teacher_quality.json
```

It marks `is_usable_teacher=false` for samples with issues such as:

- static traces
- short valid tracks
- empty required hand or target flows
- excessive NaN / Inf values
- too few queries
- missing role ids
- detected mask-fallback markers

At the current probe stage, do not treat `teacher_quality.json` as the primary success criterion. Prefer manual visual inspection first.

---

## Outputs

For a successful masked-role run, the most important artifacts are:

```text
llm_semantics.json          # task-level semantic roles
selected_traces.npy         # all selected role-aware traces
hand_flow.npy               # hand / gripper role traces
tool_flow.npy               # tool role traces
target_flow.npy             # target object role traces
query_role_id.npy           # role id for each query
query_role_name.json        # role-name mapping
teacher_targets.npz         # packaged teacher target for downstream training
```

These outputs form the bridge between semantic video understanding and self-supervised 2D/3D trajectory learning.

---

## Included Workflow Files

The current workflow code kept under `run_pipeline/code/` is:

```text
llm_to_dataset.py
dino_hoi_detector.py
traceforge/infer.py
traceforge/visualize_sample_2d_flow.py
traceforge/visualize_single_image.py
traceforge/utils/role_query_utils.py
traceforge/utils/video_depth_pose_utils.py
```

Everything else should be imported from dependency repositories, especially the official TraceForge repo.

---

## Common Pitfalls

| Symptom | Likely cause | Suggested fix |
|---|---|---|
| GroundingDINO reports `_C` is missing | CUDA extension was not built correctly | Set `CUDA_HOME`, then reinstall GroundingDINO |
| Trace output looks static | `future_len < seq_len` | Use `--future-len 128` or any value `>= 16` |
| `llm_semantics.json` contains generic `object` only | LLM was not used or semantics fallback was triggered | Check API key, provider, `semantic_debug.llm_used`, and `--semantic-mode` |
| `hand_flow.npy` / `target_flow.npy` is empty | Role masks failed or grid fallback was used | Inspect role overlays and `query_source.json` |
| `run_pipeline` cannot import TraceForge modules | External TraceForge repo is missing or not on path | Clone TraceForge under the workspace and run from the intended environment |
| First run downloads fail | Model weights or dependencies are not cached locally | Make sure outbound network access is available or pre-download weights |

---

## Citation

If this repository is useful for your research, please cite the associated paper:

```bibtex
@misc{han2026bridgeact,
  title         = {BridgeACT: Bridging Human Demonstrations to Robot Actions via Unified Tool-Target Affordances},
  author        = {Han, Yifan and Liu, Jianxiang and Zhang, Haoyu and Gu, Yuqi and Guo, Yunhan and Lian, Wenzhao},
  year          = {2026},
  eprint        = {2604.23249},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

Please also cite the upstream projects and datasets used in your experiments, including TraceForge, GroundingDINO, SAM2, and the corresponding data sources.


## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

> [!IMPORTANT]
> This license applies to the workflow code in this repository. External dependencies such as TraceForge, GroundingDINO, SAM2, model checkpoints, and datasets may use their own licenses. Please check and comply with their original license terms before redistribution or commercial use.

---

<div align="center">

**Semantic roles in. Task-relevant traces out.**

</div>
