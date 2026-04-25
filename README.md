<div align="center">

# Current Masked-Role Pipeline

**Global runbook for the current masked-role workflow**

[![Pipeline](https://img.shields.io/badge/Pipeline-Masked--Role-blue)]()
[![Core](https://img.shields.io/badge/Core-TraceForge-black)]()
[![Detection](https://img.shields.io/badge/Detection-GroundingDINO-orange)]()
[![Segmentation](https://img.shields.io/badge/Segmentation-SAM2-green)]()
[![LLM API](https://img.shields.io/badge/LLM%20API-Multi--Provider-purple)]()

</div>

---

## Overview

This runbook is the **global entrypoint** for the current masked-role workflow.

### Scope

- `TraceForge` is treated as a dependency that provides depth/pose, `prepare_query_points()`, and `PointTracker3D`
- `GroundingDINO` and `SAM2` are treated as dependency repos for online role mask generation

### Current goal

- keep the official TraceForge tracking path intact
- replace only the 2D query source with mask-constrained sampling
- preserve hand/tool/target role identity in exported flows

---

## Table of Contents

- [Overview](#overview)
- [Directory Layout](#directory-layout)
- [Dependencies](#dependencies)
  - [Required repos and paths](#required-repos-and-paths)
  - [Environment Setup](#environment-setup)
  - [CUDA Notes](#cuda-notes)
  - [Required checkpoints](#required-checkpoints)
  - [Environment requirements](#environment-requirements)
- [LLM Semantics API](#llm-semantics-api)
  - [BridgeData semantics](#bridgedata-semantics)
  - [rvideo semantics](#rvideo-semantics)
  - [Verify LLM usage](#verify-llm-usage)
- [Semantics Source](#semantics-source)
- [Mask Source](#mask-source)
- [Tracker Constraint](#tracker-constraint)
- [Recommended Parameters](#recommended-parameters)
- [Usage](#usage)
  - [Full automated pipeline](#full-automated-pipeline)
  - [Inference only on existing trajectories](#inference-only-on-existing-trajectories)
  - [Single trajectory](#single-trajectory)
  - [Included code files](#included-code-files)
- [Validation Notes](#validation-notes)
- [Validation](#validation)
- [BridgeData Branch](#bridgedata-branch)
  - [Stage Semantics](#stage-semantics)
  - [Current Validation Priority](#current-validation-priority)
- [Output Note](#output-note)

---

## Directory Layout

```text
<PATH_TO_WORKSPACE>/
  run_book/
    README.md
  run_pipeline/
    README.md
    run_traceforge_pipeline.py
    run_single_traj.py
    code/
      adapters/
        bridge_data_adapter.py
      llm_to_dataset.py
      dino_hoi_detector.py
      tools/
        export_teacher_targets.py
        filter_teacher_targets.py
      traceforge/
        infer.py
        visualize_sample_2d_flow.py
        visualize_single_image.py
        utils/
  outputs_run_pipeline/
  datasets/
```

### Notes

- `run_pipeline/` is the submission bundle for the current workflow code.
- `run_book/` is the global usage and structure reference.
- `outputs_run_pipeline/` is the global output root and stays separate from the code directory.
- `run_pipeline/` intentionally keeps only the current workflow delta.
- Original TraceForge code that is not modified for this workflow should continue to be imported from the TraceForge repo directly.

---

## Dependencies

> [!IMPORTANT]
> `run_pipeline` alone is **not enough to run**.

You still need:

- TraceForge repo for `models/` and the tracker checkpoint
- SAM2 repo for online mask segmentation
- GroundingDINO repo for text-conditioned detection
- a working `traceforge` conda environment
- a CUDA-capable GPU for practical inference
- an API key for the LLM labeling stage

Use the `traceforge` conda environment and, when already inside it, prefer direct `python ...`.

### Required repos and paths

Clone the dependency repos under a common workspace root, for example:

- TraceForge: `<PATH_TO_WORKSPACE>/TraceForge`
- SAM2: `<PATH_TO_WORKSPACE>/sam2`
- GroundingDINO: `<PATH_TO_WORKSPACE>/GroundingDINO`

Suggested clone commands:

```bash
git clone https://github.com/Yoonkyo/TraceForge.git <PATH_TO_WORKSPACE>/TraceForge
git clone https://github.com/facebookresearch/sam2.git <PATH_TO_WORKSPACE>/sam2
git clone https://github.com/idea-research/groundingdino.git <PATH_TO_WORKSPACE>/GroundingDINO
```

### Environment Setup

Recommended setup is to use the official TraceForge environment as the base, then install the mask-generation repos into the same `traceforge` environment.

```bash
cd <PATH_TO_WORKSPACE>/TraceForge
conda create -n traceforge python=3.11
conda activate traceforge
bash setup_env.sh
```

Install SAM2 in editable mode:

```bash
cd <PATH_TO_WORKSPACE>/sam2
pip install -e .
```

Install GroundingDINO in editable mode:

```bash
cd <PATH_TO_WORKSPACE>/GroundingDINO
pip install -e .
```

Install small runtime utilities used by the wrapper scripts if they are not already present after `setup_env.sh`:

```bash
pip install requests pillow opencv-python matplotlib loguru rich tqdm mediapy viser
```

> [!NOTE]
> `run_pipeline/` itself is not a Python package and does not need `pip install -e .`; run its entry scripts directly from the `traceforge` environment.

### CUDA Notes

GroundingDINO builds a CUDA extension. Before installing it, verify that `CUDA_HOME` points to the CUDA toolkit matching the PyTorch CUDA runtime:

```bash
which nvcc
echo $CUDA_HOME
```

If `CUDA_HOME` is empty and `nvcc` is under `/usr/local/cuda/bin/nvcc`, use:

```bash
export CUDA_HOME=/usr/local/cuda
```

If GroundingDINO later fails with `NameError: name '_C' is not defined`, reinstall GroundingDINO after fixing `CUDA_HOME`.

### Required checkpoints

Place checkpoints at these paths:

- TraceForge tracker checkpoint:
  `<PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth`
- SAM2 checkpoint:
  `<PATH_TO_WORKSPACE>/sam2/checkpoints/sam2.1_hiera_large.pt`
- GroundingDINO checkpoint:
  `<PATH_TO_WORKSPACE>/GroundingDINO/checkpoint/groundingdino_swint_ogc.pth`

Download examples:

```bash
mkdir -p <PATH_TO_WORKSPACE>/TraceForge/checkpoints
wget -O <PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth   https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth

mkdir -p <PATH_TO_WORKSPACE>/GroundingDINO/checkpoint
wget -O <PATH_TO_WORKSPACE>/GroundingDINO/checkpoint/groundingdino_swint_ogc.pth   https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Download the SAM2.1 Hiera-L checkpoint following the SAM2 repo instructions and place it at:

```text
<PATH_TO_WORKSPACE>/sam2/checkpoints/sam2.1_hiera_large.pt
```

### Environment requirements

- conda env name: `traceforge`
- inference is expected to run on a CUDA-capable GPU
- LLM labeling requires the provider-specific API key in the environment
- first-time model setup may require outbound network access if dependencies or Hugging Face weights are not already cached locally

If the required API key for the selected provider is missing, `--stage llm`, `--stage adapt` with LLM semantics, and `--stage all` will stop before labeling starts.

Quick import check:

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

The LLM is used only in the semantic-labeling/adaptation stage. `code/traceforge/infer.py` does not call the LLM during tracking; inference only reads the generated `llm_semantics.json`.

The API client uses the OpenAI-compatible `/chat/completions` protocol. Provider presets only fill endpoint and API-key env defaults; the model can always be overridden with `--model`.

### Built-in provider presets

- `--llm-provider moonshot`: default model `kimi-k2.5`, base URL `https://api.moonshot.cn/v1`, key env `MOONSHOT_API_KEY`
- `--llm-provider openai`: base URL `https://api.openai.com/v1`, key env `OPENAI_API_KEY`, model must be passed explicitly with `--model`
- `--llm-provider custom`: pass `--model`, `--base-url`, and `--api-key-env` explicitly

### Provider selection summary

Use the new unified interface consistently across the pipeline:

- `--llm-provider` selects the provider preset
- `--model` selects the concrete model name
- `--base-url` is only needed for `custom`, or when you want to override a preset endpoint
- `--api-key-env` is only needed for `custom`, or when you want to override the default key variable name

### Set API keys

Set the relevant key in the active `traceforge` shell before running semantic generation:

```bash
export MOONSHOT_API_KEY=<YOUR_MOONSHOT_API_KEY>
# or
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

### Multi-model usage

The current API path supports **multiple models**, as long as they are reachable through the selected OpenAI-compatible provider configuration.

Examples:

- Moonshot/Kimi via preset:
  - `--llm-provider moonshot --model kimi-k2.5`
  - `--llm-provider moonshot --model <OTHER_SUPPORTED_MOONSHOT_MODEL>`
- OpenAI via preset:
  - `--llm-provider openai --model <OPENAI_VISION_MODEL>`
- Custom OpenAI-compatible endpoint:
  - `--llm-provider custom --base-url <YOUR_BASE_URL> --api-key-env <YOUR_KEY_ENV> --model <YOUR_MODEL_NAME>`

So the effective behavior is:

- **provider preset** decides the default endpoint and which environment variable stores the API key
- **`--model`** decides which concrete model is called
- **`custom`** lets you plug in any OpenAI-compatible service, as long as it supports the needed multimodal/chat completion path

### BridgeData semantics

BridgeData semantics are generated by `code/adapters/bridge_semantics.py` through `bridge_data_adapter.py`. The adapter first extracts rule candidates from task/folder/meta/frame context, then optionally calls the LLM to standardize:

- `core_description`
- `action`
- `tool`
- `target`
- `interaction`
- `folder_label`

Recommended probe command:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py   --source-mode bridge_raw   --stage adapt   --bridge-root <PATH_TO_BRIDGEDATA_RAW>   --dataset-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes   --task-filter pnp_utensils   --max-samples-per-task 5   --max-frames-per-video 16   --llm-provider moonshot   --semantic-mode llm   --model kimi-k2.5   --semantic-timeout-sec 180   --semantic-retries 3   --overwrite-llm
```

OpenAI-compatible example:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py   --source-mode bridge_raw   --stage adapt   --bridge-root <PATH_TO_BRIDGEDATA_RAW>   --dataset-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes   --task-filter pnp_utensils   --max-samples-per-task 5   --max-frames-per-video 16   --llm-provider openai   --model <OPENAI_VISION_MODEL>   --semantic-mode llm   --overwrite-llm
```

Custom provider example:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py   --source-mode bridge_raw   --stage adapt   --bridge-root <PATH_TO_BRIDGEDATA_RAW>   --dataset-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes   --task-filter pnp_utensils   --max-samples-per-task 5   --max-frames-per-video 16   --llm-provider custom   --base-url <YOUR_BASE_URL>   --api-key-env <YOUR_KEY_ENV>   --model <YOUR_MODEL_NAME>   --semantic-mode llm   --overwrite-llm
```

Semantic modes:

- `--semantic-mode llm`: require the API call to succeed; fail if the LLM times out or errors.
- `--semantic-mode rules_then_llm`: try the API, then fall back to rule candidates if the API fails.
- `--semantic-mode rules`: do not call the API; use only deterministic rule extraction.

For BridgeData, the LLM request includes task/meta context plus a small set of sampled full RGB frames encoded as image inputs. The default path is one episode at a time, not one giant batch request.

### rvideo semantics

For the original `rvideo` source mode, `code/llm_to_dataset.py` calls the same OpenAI-compatible API and writes `llm_semantics.json` while exporting trajectories:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py   --source-mode rvideo   --stage llm   --video-root <PATH_TO_WORKSPACE>/datasets/video   --dataset-root <PATH_TO_WORKSPACE>/datasets   --task-filter close_oven   --frames-per-video 6   --max-samples-per-task 5   --llm-provider moonshot   --model kimi-k2.5   --overwrite-llm
```

OpenAI-compatible example:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py   --source-mode rvideo   --stage llm   --video-root <PATH_TO_WORKSPACE>/datasets/video   --dataset-root <PATH_TO_WORKSPACE>/datasets   --task-filter close_oven   --frames-per-video 6   --max-samples-per-task 5   --llm-provider openai   --model <OPENAI_VISION_MODEL>   --overwrite-llm
```

### Verify LLM usage

After `--stage adapt` or `--stage llm`, inspect one generated file:

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

For BridgeData outputs, `semantic_debug.llm_used` should be `true` when the LLM API was actually used. The file should contain concrete `tool` and `target` values, for example `gripper` and `spatula`, not only generic defaults such as `object`.

---

## Semantics Source

`code/traceforge/infer.py` does not call an LLM during inference.

Priority:

1. read `<traj_dir>/llm_semantics.json`
2. if missing, parse the parent task folder name

The automated path is therefore:

1. `llm_to_dataset.py` exports dataset-like trajectories and writes `llm_semantics.json`
2. TraceForge masked-role inference reads that file later

---

## Mask Source

For `masked_roles`, masks are generated online:

1. `GroundingDINO` predicts role boxes from text prompts
2. `SAM2` segments masks from those boxes
3. query points are sampled only from the foreground pixels in those masks

There is no fallback to full-image grid queries when role masks fail.

---

## Tracker Constraint

Current checkpoint:

- `<PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth`
- model class: `PointTracker3D`
- `seq_len = 16`

Required:

- `future_len >= model.seq_len`

If `future_len < seq_len`, the tracker does not enter a tracking window and outputs static coordinates.

---

## Recommended Parameters

- `--input_layout rvideo_traj_dataset`
- `--query_mode masked_roles`
- `--fps 1`
- `--max_num_frames 2000`
- `--future_len 128`
- `--frame_drop_rate 4`

---

## Usage

### Full automated pipeline

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py   --stage all   --video-root <PATH_TO_WORKSPACE>/datasets/video   --dataset-root <PATH_TO_WORKSPACE>/datasets   --traceforge-output <PATH_TO_WORKSPACE>/outputs_run_pipeline/traceforge_auto   --checkpoint <PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth   --query-mode masked_roles   --fps 1   --future-len 128   --frame-drop-rate 4
```

### Inference only on existing trajectories

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py   --stage infer   --dataset-root <PATH_TO_WORKSPACE>/datasets   --traceforge-output <PATH_TO_WORKSPACE>/outputs_run_pipeline/traceforge_auto   --checkpoint <PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth   --task-filter close_oven   --query-mode masked_roles
```

### Single trajectory

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_single_traj.py   --traj-dir <PATH_TO_WORKSPACE>/datasets/close_oven/traj_001   --out-dir <PATH_TO_WORKSPACE>/outputs_run_pipeline/traceforge_formal   --checkpoint <PATH_TO_WORKSPACE>/TraceForge/checkpoints/tapip3d_final.pth
```

### Included code files

The current workflow code kept under `run_pipeline/code/` is:

- `llm_to_dataset.py`
- `dino_hoi_detector.py`
- `traceforge/infer.py`
- `traceforge/visualize_sample_2d_flow.py`
- `traceforge/visualize_single_image.py`
- `traceforge/utils/role_query_utils.py`
- `traceforge/utils/video_depth_pose_utils.py`

Everything else should be imported from the dependency repos, especially `TraceForge`.

---

## Validation Notes

- `run_pipeline` is intended to contain only the workflow delta, not a copied dependency closure
- the bundled scripts are expected to run against the external dependency repos listed above
- for a first full run, make sure the required API key, GPU environment, checkpoints, and any needed online model downloads are available

---

## Validation

Prefer:

1. inspect `llm_semantics.json`
2. inspect `selected_traces.npy`, `hand_flow.npy`, `tool_flow.npy`, `target_flow.npy`
3. run the TraceForge 2D and 3D checkers
4. use the bundled 2D overlay script when needed

Do not use the official single-sample viewer as the only success criterion for `masked_roles`.

Checker commands:

```bash
python <PATH_TO_WORKSPACE>/TraceForge/checker/batch_process_result_checker.py   <PATH_TO_WORKSPACE>/outputs_run_pipeline/traceforge_auto

python <PATH_TO_WORKSPACE>/TraceForge/checker/batch_process_result_checker_3d.py   <PATH_TO_WORKSPACE>/outputs_run_pipeline/traceforge_auto
```

---

## BridgeData Branch

This branch is for self-supervised data generation, not policy learning. BridgeData supplies raw robot videos and task distribution; TraceForge generates automatic 3D trace targets used later for TraceGen/world-model/trace-prediction pretraining. Action/state arrays are preserved by the adapter for later weak supervision or downstream fine-tuning, but this branch does not connect an execution layer.

BridgeData-specific reading is isolated in the adapter. The adapter also writes `llm_semantics.json`; it uses the configured OpenAI-compatible API through the unified provider interface, not only rules:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/code/adapters/bridge_data_adapter.py   --bridge-root <PATH_TO_BRIDGEDATA_RAW>   --output-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes   --semantic-mode llm   --llm-provider moonshot   --model kimi-k2.5
```

The adapter exports only the local episode format consumed by the existing TraceForge workflow:

```text
bridge_episodes/<task_label>/traj_000000/
  rgb_0.png
  rgb_1.png
  meta.json
  action.npy
  state.npy
  language.txt
  llm_semantics.json
  camera_in.npy  # optional, only if present in the source
```

> [!NOTE]
> The examples below follow the **new unified API interface**. Prefer `--llm-provider` + `--model` over the older split flags such as `--semantic-model`, `--semantic-base-url`, and `--semantic-api-key-env`.

Preferred entrypoint is now the unified pipeline with `--source-mode bridge_raw`:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py   --source-mode bridge_raw   --stage adapt   --bridge-root <PATH_TO_BRIDGEDATA_RAW>   --dataset-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes   --task-filter pnp_utensils   --max-samples-per-task 5   --max-frames-per-video 16   --overwrite-llm
```

### Stage Semantics

`--stage adapt` only runs:

- BridgeData raw JPEG/PNG/pkl scan
- episode export to `rvideo_traj_dataset` layout
- LLM semantics generation into `llm_semantics.json`

It does not run TraceForge inference. After changing or regenerating `llm_semantics.json`, rerun `--stage infer`; existing TraceForge output directories are not automatically updated.

`--stage infer` only runs masked-role TraceForge on the current adapted episodes:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py   --source-mode bridge_raw   --stage infer   --dataset-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes   --traceforge-output <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_traceforge   --query-mode masked_roles   --fps 1   --future-len 128   --frame-drop-rate 4   --max-num-frames 2000
```

`--stage all` runs adapt then infer in one command. Use it only when you intentionally want to regenerate semantics and then immediately run TraceForge.

BridgeData inference uses the same masked-role TraceForge mainline defaults:

- `--input_layout rvideo_traj_dataset`
- `--query_mode masked_roles`
- `--fps 1`
- `--max_num_frames 2000`
- `--future_len 128`
- `--frame_drop_rate 4`

By default, missing role masks still fail fast. For BridgeData probe runs where keeping a sample is more important than strict role-only queries, opt into per-frame grid fallback:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/run_traceforge_pipeline.py   --source-mode bridge_raw   --stage infer   --dataset-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes   --traceforge-output <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_traceforge   --query-mode masked_roles   --masked_role_fallback grid
```

Fallback is recorded per query frame in `role_queries/<episode>_<frame>_query_source.json` as `masked_roles` or `fallback_grid`. Grid-fallback queries use role id `0` (`generic`), so they will not appear in `hand_flow.npy`, `tool_flow.npy`, or `target_flow.npy`; inspect `selected_traces.npy` and the query source JSON for those frames.

Default outputs are separate from the current mainline:

- adapted episodes: `<PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes`
- TraceForge outputs: `<PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_traceforge`
- training assets: `<PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_teacher_targets`

Each training-asset episode directory contains `teacher_targets.npz` with at least:

- `trace_3d`
- `trace_2d`
- `valid_steps`
- `visibility`
- `query_frame_index`
- `role_id`

It also copies the stable role/flow files for inspection and downstream use:

- `selected_traces.npy`
- `hand_flow.npy`
- `tool_flow.npy`
- `target_flow.npy`
- `query_role_id.npy`
- `query_role_name.json`

### Current Validation Priority

At the current probe stage, do not treat `teacher_quality.json` as the primary success criterion. Prefer manual visual inspection:

1. inspect `llm_semantics.json` and confirm `semantic_debug.llm_used=true`
2. inspect adapter preview images and semantics
3. inspect role/query overlays
4. inspect `selected_traces.npy`, `hand_flow.npy`, `tool_flow.npy`, `target_flow.npy`
5. run TraceForge 2D/3D checkers
6. use the single-image 3D Viser viewer for representative samples

Batch visualizations from existing TraceForge outputs:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/code/traceforge/visualize_sample_2d_flow.py   --batch-output-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_traceforge   --adapter-root <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_episodes   --vis-output-dir <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_visualizations
```

Quality filtering is optional at this stage and is intended for later large-scale teacher-target generation:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/code/tools/filter_teacher_targets.py   --teacher-output <PATH_TO_WORKSPACE>/outputs_run_pipeline/bridge_teacher_targets
```

The filter writes `teacher_quality.json` and marks `is_usable_teacher=false` for samples with static traces, short valid tracks, empty required hand/target flows, excessive NaN/Inf, too few queries, missing role ids, or detected mask-fallback markers. The masked-role path still must fail rather than falling back to full-image grid queries by default.

2D overlay command:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/code/traceforge/visualize_sample_2d_flow.py   --sample-npz <PATH_TO_EPISODE_DIR>/samples/<EPISODE_NAME>_<QUERY_FRAME>.npz   --episode-dir <PATH_TO_EPISODE_DIR>   --output <PATH_TO_EPISODE_DIR>/vis_2d/<EPISODE_NAME>_<QUERY_FRAME>.mp4
```

Single-image 3D viewer command:

```bash
python <PATH_TO_WORKSPACE>/run_pipeline/code/traceforge/visualize_single_image.py   --npz_path <PATH_TO_EPISODE_DIR>/samples/<EPISODE_NAME>_<QUERY_FRAME>.npz   --image_path <PATH_TO_EPISODE_DIR>/images/<EPISODE_NAME>_<QUERY_FRAME>.png   --depth_path <PATH_TO_EPISODE_DIR>/depth/<EPISODE_NAME>_<QUERY_FRAME>.png   --port 8080
```

---
