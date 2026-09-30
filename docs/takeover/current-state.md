# Current-state inventory

Snapshot date: 2026-09-30.

This file distinguishes facts verified in the repository from external runtime state that
could not be observed.

## Verified repository state

- The package is `text-to-anime` version `0.1.0`, Python `>=3.10`.
- Source is under `src/text_to_anime`; 21 console scripts are declared in
  `pyproject.toml`.
- Checked-in configs cover AniSora T2V, Wan2.2 I2V, Wan2.2 Anita production high/low and
  overfit experiments, VACE 1.3B production, and the standalone production U-Net.
- The test tree contains unit coverage for manifests, caption construction, AnimeShooter,
  Anita layout/splitting, video resizing, task balancing, Wan2.2 conditioning/render
  metrics, VACE timeline/conditioning/loss weights, and background filling.
- No `docs/` directory existed before this documentation set.
- No Git repository metadata is present in this workspace copy; `git status` and
  `git log` report that it is not a Git repository.
- No CI workflow/configuration was found in the repository inventory.
- The checked-in `.venv` contains runtime package entry points but no `pytest` executable.
  The system shell also has no `pytest`; therefore the suite could not be run in the
  inspected environment without installing development dependencies.

## Not verified from this repository

- Existence, integrity, or exact revision of any checkpoint under
  `/data/shasegawa/t2a/models`.
- Existence, license record, completeness, counts, or hashes of external datasets.
- Contents or current validity of manifests under `/data/shasegawa/t2a/manifests`.
- Whether any training job is active or completed.
- Health or quality of LoRAs and renders under `/data/shasegawa/t2a/outputs`.
- CUDA driver/runtime versions, available GPUs, Accelerate configuration, and free disk.
- Availability of upstream AniSora source code referenced by smoke/web scripts.
- OpenAI credentials or whether external captioning is authorized.

## Implemented paths and maturity assessment

| Path | Implementation status | Validation support | Takeover assessment |
|---|---|---|---|
| Generic Wan/AniSora T2V LoRA | Complete trainer + inference | periodic prompt videos | usable research path; needs quantitative validation |
| Wan2.2 first-frame I2V LoRA | Complete trainer + inference | 50–100 fixed renders | usable research path; expensive validation volume |
| Wan2.2 production LoRA | Trainer + custom dual-expert renderer | external overfit/render protocol | promising but validation is operator-driven |
| Wan VACE Anita LoRA | Trainer + stock-pipeline renderer | fixed per-task losses and held-out renders | most aligned production experiment |
| Standalone 3D U-Net | Train/infer baseline | visual grids only | experimental baseline; split can leak related shots |
| Native AniSora web wrapper | Simple local job UI | subprocess log/status | operational convenience, not hardened service |

## Current configuration facts

- Default external root is `/data/shasegawa/t2a`.
- The VACE config uses Wan2.1 VACE 1.3B, 832×480, up to 33 frames, effective 12 fps,
  LoRA rank 32, and three processes in the documented launch.
- Balanced Wan2.2 production configs use separate 512×288, 33-frame, 8 fps high/low
  runs with rank 32 and a task mixture of 1:1:2.
- The production manifest comment records an observed task imbalance of 306 line-art,
  183 character-color, and 39 compose/refine records. Treat those numbers as historical
  config context until regenerated and verified against the current dataset.
- The standard I2V config uses 640×360, 49 frames at 12 fps and a high-noise adapter.
- The AniSora T2V config expects a Diffusers-compatible conversion at
  `/data/shasegawa/t2a/models/Index-anisora-diffusers/V3.2`, not the native download.

## Known code/operations gaps

1. No unified preflight or asset checksum mechanism.
2. No full-state resume for Accelerate/LoRA trainers.
3. No integrated validation loop in Wan2.2 production training.
4. Hard-coded absolute defaults reduce portability.
5. No common run registry tying config, manifest hash, base model revision, code revision,
   and evaluation decision together.
6. No CI baseline in this snapshot, and tests are not runnable in the existing venv.
7. Generic T2V/I2V validation writes many videos but no structured metric log.
8. The local HTTP server has process-local job state and locking only; it has no auth,
   durable queue, multi-worker coordination, or production deployment controls.

## Immediate verification commands for the new owner

After installing development dependencies:

```bash
pytest -q
ruff check src tests
```

Then inventory external state without changing it:

```bash
find /data/shasegawa/t2a/models -maxdepth 2 -type f -name config.json -print
find /data/shasegawa/t2a/manifests -maxdepth 1 -type f -name '*.jsonl' -print
find /data/shasegawa/t2a/outputs -maxdepth 3 -type f \
  \( -name training_args.json -o -name training_metadata.json -o -name render_summary.json \) -print
nvidia-smi
df -h /data/shasegawa/t2a
```

Do not regenerate manifests or launch training until the existing external assets and
outputs are inventoried; regeneration can change record selection, captions, and splits.
