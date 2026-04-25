Current workflow code lives in this directory.

Global usage and structure documentation:

- `/home/zhy/data/run_book/README.md`

Global outputs root:

- `/home/zhy/data/outputs_run_pipeline`

BridgeData self-supervised teacher-target branch:

- adapter: `/home/zhy/data/run_pipeline/code/adapters/bridge_data_adapter.py`
- unified entrypoint: `/home/zhy/data/run_pipeline/run_traceforge_pipeline.py --source-mode bridge_raw`
- teacher export: `/home/zhy/data/run_pipeline/code/tools/export_teacher_targets.py`
- quality filter: `/home/zhy/data/run_pipeline/code/tools/filter_teacher_targets.py`

Default BridgeData branch outputs stay separate from the current mainline:

- adapted episodes: `/home/zhy/data/outputs_run_pipeline/bridge_episodes`
- raw TraceForge outputs: `/home/zhy/data/outputs_run_pipeline/bridge_traceforge`
- training assets: `/home/zhy/data/outputs_run_pipeline/bridge_teacher_targets`
