# Takeover guide

## Executive summary

This repository is an active research toolkit, not a packaged production service. Its
strongest architectural direction is Wan VACE LoRA training for Anita production stages,
because training and inference share VACE-native control/mask/reference conditioning. The
Wan2.2 two-expert source-shot path is the main alternative and includes better custom
render diagnostics, but it requires separately trained high/low adapters and a custom
sampler. Generic T2V and first-frame I2V are independent capabilities, not prerequisites
for VACE.

The repository alone is insufficient to reproduce prior results: model weights, datasets,
manifests, Accelerate configuration, and output artifacts live under an external `/data`
tree. Obtain an inventory or copy of that tree before assuming any historical experiment
is recoverable.

## First-day reading order

1. Read [current-state.md](current-state.md) for verified snapshot facts.
2. Read [../architecture.md](../architecture.md), especially the model-path boundaries.
3. Read [../training-methodology.md](../training-methodology.md) before changing task
   conditioning or evaluation.
4. Follow [runbook.md](runbook.md) to verify the environment and run a CPU-only test suite.
5. Inspect the active experiment's `training_args.json`, `training_metadata.json`, split,
   and render summary under `/data/shasegawa/t2a/outputs`.

## What is canonical

- Source code and CLI declarations: `src/text_to_anime/` and `pyproject.toml`.
- Experiment settings: checked-in YAML plus the output run's resolved
  `training_args.json`. The resolved file wins when reconstructing a historical run.
- Dataset identity: the exact manifest, background-plate index, and split emitted for a
  run. A filename alone is not enough; archive or checksum it.
- Model identity: exact local checkpoint directory/revision and whether it is native
  AniSora or Diffusers-compatible.
- Evaluation decision: render settings, seeds, per-sample JSON, comparison MP4s, and human
  notes. Training loss alone is not canonical evidence.

## Recommended path for new production work

Use `configs/wan21_vace_1.3b_anita_480p_lora.yaml` to settle task definitions and data
quality cheaply. Run the base VACE renderer first, then a tiny LoRA overfit, then held-out
training. Move to a larger VACE checkpoint only after the 1.3B experiment demonstrates
that all three transformations work without copying.

Keep the Wan2.2 balanced high/low configs as a comparison track. Do not mix a high adapter
from one config/dataset with a low adapter from another without an explicit ablation.

## Principal risks

### Reproducibility

- External data and outputs are not versioned here.
- Absolute paths are embedded throughout configs.
- LoRA checkpoints cannot resume optimizer/scheduler state.
- The inspected workspace has no Git metadata, so commit provenance cannot be established
  from this copy.

### Validation gaps

- Wan2.2 production training has no built-in validation.
- I2V/T2V validation renders are useful but do not calculate quality metrics.
- The standalone U-Net uses record-level random splitting and can leak related shots.
- There is no CI configuration in this snapshot.

### Infrastructure

- The documented host has broken/silent CUDA peer copies. Preserve CPU-staged transfers
  and `NCCL_P2P_DISABLE=1` until a targeted peer-copy test proves they are unnecessary.
- Training requires large checkpoints and GPU memory; import/unit tests should remain
  runnable without downloading those assets.
- `ffmpeg` is a runtime dependency but is not managed by `pyproject.toml`.

### Data and rights

- AnimeShooter/Sakuga-derived data may have research-only or unclear commercial rights.
- Anita is treated as licensed auxiliary data, but the operator must retain the actual
  license/provenance record.
- OpenAI captioning transmits selected frames to an external service; use it only when
  data policy permits and never store API keys in manifests or logs.

## Suggested maintenance priorities

1. Recover version-control history and write down the exact external asset inventory.
2. Install development dependencies and make the current unit suite a CI gate.
3. Add a read-only preflight command that validates checkpoints, executables, manifests,
   task counts, split leakage, disk space, and GPU topology before launch.
4. Add full Accelerate checkpoint/resume for LoRA training.
5. Integrate deterministic held-out validation into Wan2.2 production training.
6. Replace hard-coded `/data/shasegawa/t2a` defaults with one consistently resolved root.
7. Add manifest/run checksums and environment/package snapshots to every output directory.
8. Define quantitative and human acceptance thresholds per task, then preserve a fixed
   golden render set.

## Safe change boundaries

Changes to common latent normalization, mask temporal packing, scheduler boundary logic,
or prompt encoding affect multiple trainers. Require unit tests plus a base/adapted render
comparison for those changes. Dataset-only adapters and CLIs are more isolated, but source
split semantics and held-frame timing must remain stable.

Never delete or overwrite `/data` outputs during cleanup without an explicit retention
decision. New experiments should use a new output directory, not reuse an existing run.

## Handoff checklist

- [ ] Repository origin and commit/tag identified.
- [ ] Python/CUDA/driver/ffmpeg/package versions captured.
- [ ] External model paths and checksums captured.
- [ ] Dataset licenses, roots, counts, and manifest hashes captured.
- [ ] Active experiment, owner, hypothesis, config, and expected completion recorded.
- [ ] Accelerate topology and `CUDA_VISIBLE_DEVICES` recorded.
- [ ] Training and validation logs reviewed for the last healthy step.
- [ ] At least one inference/render command reproduced from saved artifacts.
- [ ] Open risks and next decision written in the run directory.
- [ ] API tokens and credentials transferred through an approved secret channel only.
