Current workflow code lives in this directory.

Global usage and structure documentation:

- `run_book/README.md`

Global outputs root:

- `outputs_run_pipeline`

BridgeData self-supervised teacher-target branch:

- adapter: `run_pipeline/code/adapters/bridge_data_adapter.py`
- unified entrypoint: `run_pipeline/run_traceforge_pipeline.py --source-mode bridge_raw`
- teacher export: `run_pipeline/code/tools/export_teacher_targets.py`
- quality filter: `run_pipeline/code/tools/filter_teacher_targets.py`

Default BridgeData branch outputs stay separate from the current mainline:

- adapted episodes: `outputs_run_pipeline/bridge_episodes`
- raw TraceForge outputs: `outputs_run_pipeline/bridge_traceforge`
- training assets: `outputs_run_pipeline/bridge_teacher_targets`
