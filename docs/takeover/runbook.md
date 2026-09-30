# Operations runbook

## 1. Bootstrap

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[video,dev]'
export HF_HOME=/data/shasegawa/t2a/hf-cache
accelerate config
```

The checked-in environment in this snapshot was created with Python 3.14, while the
package declares Python 3.10+. For GPU work, use the Python and Torch/CUDA combination
validated by the target Diffusers checkpoint rather than assuming the existing venv is
reusable.

System requirements:

```bash
ffmpeg -version
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
python -c 'import diffusers, accelerate, peft; print(diffusers.__version__, accelerate.__version__, peft.__version__)'
```

On the documented multi-GPU host:

```bash
export NCCL_P2P_DISABLE=1
```

Do not remove this until an explicit GPU peer-transfer correctness test succeeds.

## 2. Verify the code baseline

```bash
ruff check src tests
pytest -q
```

The unit suite is designed around small synthetic images/tensors and should not require
model downloads or GPUs. If collection imports optional heavy packages unexpectedly,
fix the import boundary instead of provisioning foundation checkpoints for CI.

## 3. Prepare the external layout

```bash
export T2A_ROOT=/data/shasegawa/t2a
scripts/prepare_first_run.sh
```

Confirm free space before downloads and runs:

```bash
df -h /data/shasegawa/t2a
du -sh /data/shasegawa/t2a/models /data/shasegawa/t2a/datasets /data/shasegawa/t2a/outputs
```

Keep native AniSora and Diffusers-converted checkpoints in distinct directories. Trainers
require the Diffusers form; native smoke scripts require upstream AniSora code and layout.

## 4. Data workflows

### AnimeShooter T2V

```bash
scripts/download_animeshooter.sh
scripts/download_animeshooter_sources.sh

t2a-animeshooter-manifest \
  --annotations /data/shasegawa/t2a/datasets/animeshooter/raw/dataset_anime_shooter.zip \
  --videos-root /data/shasegawa/t2a/datasets/animeshooter/source_videos \
  --clips-root /data/shasegawa/t2a/datasets/animeshooter/clips \
  --output /data/shasegawa/t2a/manifests/animeshooter_raw.jsonl

t2a-extract-clips \
  --manifest /data/shasegawa/t2a/manifests/animeshooter_raw.jsonl \
  --output-manifest /data/shasegawa/t2a/manifests/animeshooter_clipped.jsonl

t2a-build-subset \
  --manifest /data/shasegawa/t2a/manifests/animeshooter_clipped.jsonl \
  --output /data/shasegawa/t2a/manifests/animeshooter_vn_motion_30k.jsonl \
  --target-size 30000

t2a-build-manifest split \
  --manifest /data/shasegawa/t2a/manifests/animeshooter_vn_motion_30k.jsonl \
  --train-out /data/shasegawa/t2a/manifests/train.jsonl \
  --val-out /data/shasegawa/t2a/manifests/val.jsonl \
  --val-ratio 0.02
```

Before training:

```bash
t2a-build-manifest validate \
  --manifest /data/shasegawa/t2a/manifests/train.jsonl \
  --require-files
```

### Anita production manifest and captions

```bash
t2a-anita-production manifest \
  --root /data/shasegawa/t2a/datasets/anita \
  --output /data/shasegawa/t2a/manifests/anita_production.jsonl \
  --max-frames-per-shot 49 \
  --min-frames-per-shot 8 \
  --shot-split-strategy semantic \
  --semantic-split-search-radius 3
```

Caption only if the data policy permits external processing:

```bash
export OPENAI_API_KEY='set-through-the-approved-secret-store'
t2a-caption-anita-shots \
  --input /data/shasegawa/t2a/manifests/anita_production.jsonl \
  --output /data/shasegawa/t2a/manifests/anita_production_captioned.jsonl \
  --provider openai \
  --model gpt-5.6 \
  --num-frames 3 \
  --image-detail low
```

For VACE compose/refine, build plates before launch:

```bash
t2a-anita-bg-plates
```

Review plate logs for unexpected static/moving classification and low coverage. Visually
inspect representative `plate.png`, `plate_coverage.png`, and per-frame backgrounds.

## 5. Launch training

Always create a new output path in the YAML or with `--output-dir`. Save the launch command
beside the run artifacts.

### Generic AniSora T2V

```bash
accelerate launch --num_processes 2 -m text_to_anime.train_wan_lora \
  --config configs/anisora_v32_animeshooter_360p_lora.yaml \
  --manifest /data/shasegawa/t2a/manifests/train.jsonl \
  --validation-manifest /data/shasegawa/t2a/manifests/val.jsonl
```

### Anita first-frame I2V

```bash
scripts/train_anita_i2v_gpus_2_3.sh
```

### Wan2.2 source-shot production

First run the three-scene overfit configs, one expert per GPU:

```bash
CUDA_VISIBLE_DEVICES=2 NCCL_P2P_DISABLE=1 \
  t2a-train-wan22-anita-production \
  --config configs/wan22_anita_production_288p_overfit3_high.yaml

CUDA_VISIBLE_DEVICES=3 NCCL_P2P_DISABLE=1 \
  t2a-train-wan22-anita-production \
  --config configs/wan22_anita_production_288p_overfit3_low.yaml
```

Then render both experts together:

```bash
t2a-render-wan22-anita-production \
  --data /data/shasegawa/t2a/manifests/anita_production_captioned.jsonl \
  --output-dir /data/shasegawa/t2a/outputs/renders/overfit3 \
  --high-lora /data/shasegawa/t2a/outputs/wan22-anita-production-overfit3-288x512-high \
  --low-lora /data/shasegawa/t2a/outputs/wan22-anita-production-overfit3-288x512-low \
  --scenes 119_a_part000,221_a_part000,204_a_part000 \
  --high-device cuda:2 --low-device cuda:3
```

Do not start balanced full runs until the overfit renders perform all three mappings.

### VACE production

```bash
scripts/download_wan21_vace.sh 1.3B
CUDA_VISIBLE_DEVICES=0,2,3 NCCL_P2P_DISABLE=1 \
  accelerate launch --num_processes 3 --mixed_precision bf16 \
  -m text_to_anime.train_wan_vace_anita \
  --config configs/wan21_vace_1.3b_anita_480p_lora.yaml
```

Render both a base-model baseline and the adapter on the same split:

```bash
t2a-render-wan-vace-anita \
  --config configs/wan21_vace_1.3b_anita_480p_lora.yaml \
  --output-dir /data/shasegawa/t2a/outputs/renders/vace-base-val \
  --device cuda:0

t2a-render-wan-vace-anita \
  --config configs/wan21_vace_1.3b_anita_480p_lora.yaml \
  --lora /data/shasegawa/t2a/outputs/wan21-vace-1.3b-anita-480p-r1 \
  --output-dir /data/shasegawa/t2a/outputs/renders/vace-r1-val \
  --device cuda:0
```

## 6. Monitor a run

Check all of the following, not only GPU utilization:

- process count matches Accelerate configuration;
- each process has stable GPU memory and nonzero utilization;
- loss and learning rate continue to be written;
- per-task sampling/loss is present for VACE;
- `checkpoint-N` directories appear at the configured interval;
- validation does not stall indefinitely or exhaust memory;
- disk capacity covers remaining checkpoints and videos.

VACE emits `train_log.jsonl` every ten synchronized steps and
`validation_log.jsonl` at validation intervals. It also writes `split.json`. Generic LoRA
runs primarily expose the progress stream and validation videos.

## 7. Evaluate and archive

For every accepted or rejected experiment, retain:

```text
training_args.json
training_metadata.json             # LoRA runs
split.json                         # VACE
train_log.jsonl / validation_log.jsonl when available
pytorch_lora_weights.safetensors   # name may be Diffusers-version dependent
render_summary.json
comparison videos and per-render JSON
source manifest/config checksums
launch command and environment versions
decision notes
```

Use teacher-forced renders first. Use chained Wan2.2 renders only to measure end-to-end
error after individual stages pass. Compare identical seeds, scheduler settings, CFG,
flow shift, dimensions, and frame counts between base and adapted runs.

## 8. Failure handling

### CUDA OOM

Keep batch size one. Reduce resolution or frame count to another valid `4k+1` length,
enable gradient checkpointing, use VAE tiling where compatible, or increase accumulation
instead of batch size. Record the changed effective batch and learning-rate decision.

### Zero/corrupt cross-GPU tensors

Stop the run. Confirm `NCCL_P2P_DISABLE=1`, avoid direct device-to-device copies, and use
the renderer's CPU-staged transfer pattern. Do not treat a run with silent zeros as valid.

### Model copies its condition

Compare `psnr_gen_vs_condition` with `psnr_condition_vs_target`. Run the overfit test,
inspect change-weighted loss settings, task balance, masks, and whether the intended task
records actually contain distinct source/target frames.

### No color or low saturation

Inspect color reference selection, task prompts, source/target alignment, and per-task
sampling. Confirm character-color target PNGs are composited over white as expected and
the reference is outside the sampled clip when possible.

### Resume after interruption

LoRA trainers currently have no true resume function. `checkpoint-N` contains adapter
weights but not optimizer, scheduler, sampler, or global-step state. Preserve it as a
partial artifact, then start a clearly named new run. The standalone U-Net checkpoint does
include optimizer state, but its trainer also lacks a CLI resume path.

### Validation worsens while training loss improves

Stop scaling. Check split leakage, task mixture, copying metrics, and qualitative panels.
Choose the best validated adapter rather than automatically using the final directory.
