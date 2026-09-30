# Text-to-Anime

This repo contains practical training and inference code for anime text-to-video domain adaptation. It is built around a Diffusers-compatible Wan/AniSora-style backbone and trains LoRA adapters rather than attempting full text-to-video pretraining.

The intended first milestone is narrow:

```text
one character, one background, one continuous action, one camera instruction, 3-5 seconds
```

## What is included

- JSONL manifest tooling for filtered AnimeShooter/Sakuga-style clips.
- A structured caption helper for later motion-aware captions. The initial run uses AnimeShooter's existing narrative/descriptive captions.
- Video loading that samples fixed-length clips while preserving encoded held frames.
- Wan text-to-video LoRA training with Accelerate.
- Wan text-to-video inference with optional LoRA loading.
- Concrete AniSora V3.2/AnimeShooter path configs and a starter 360p two-L40S config.


## First Run: AniSora V3.2 + AnimeShooter

Prepare the large-file layout under `/data/shasegawa/t2a`:

```bash
scripts/prepare_first_run.sh
```

Download AniSora V3.2 native weights to `/data/shasegawa/t2a/models/Index-anisora/V3.2`:

```bash
export HF_HOME=/data/shasegawa/t2a/hf-cache
scripts/download_anisora_v32.sh
```

Download AnimeShooter metadata/files to `/data/shasegawa/t2a/datasets/animeshooter/raw`:

```bash
export HF_HOME=/data/shasegawa/t2a/hf-cache
scripts/download_animeshooter.sh
```

Convert AnimeShooter shot annotations to the trainer manifest, using AnimeShooter's own narrative/descriptive captions:

```bash
t2a-animeshooter-manifest \
  --annotations /data/shasegawa/t2a/datasets/animeshooter/raw/dataset_anime_shooter.zip \
  --videos-root /data/shasegawa/t2a/datasets/animeshooter/source_videos \
  --clips-root /data/shasegawa/t2a/datasets/animeshooter/clips \
  --output /data/shasegawa/t2a/manifests/animeshooter_raw.jsonl
```

If `video_ids.txt` is present and `yt-dlp` is installed, download source videos and extract shot clips:

```bash
scripts/download_animeshooter_sources.sh

t2a-extract-clips \
  --manifest /data/shasegawa/t2a/manifests/animeshooter_raw.jsonl \
  --output-manifest /data/shasegawa/t2a/manifests/animeshooter_clipped.jsonl
```

Build the initial high-quality VN-weighted subset without recaptioning:

```bash
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

Training uses `configs/anisora_v32_animeshooter_360p_lora.yaml`. That config points to `/data/shasegawa/t2a/models/Index-anisora-diffusers/V3.2`, meaning AniSora V3.2 must be available as a `WanPipeline`-compatible Diffusers folder for this trainer. Keep the native checkpoint at `/data/shasegawa/t2a/models/Index-anisora/V3.2` for upstream AniSora inference and conversion work.


## Optional: AnitaDataset

AnitaDataset is useful as a licensed auxiliary animation-style source, not as the main captioned T2V dataset. It provides 1080p image sequences for sketch, color, and composition folders, but no captions or semantic annotations.

Download the official Google Drive archive:

```bash
scripts/download_anita.sh
unzip -q /data/shasegawa/t2a/datasets/anita/raw/Anita_Dataset.zip -d /data/shasegawa/t2a/datasets/anita
```

Build or render an auxiliary manifest from the extracted image sequences:

```bash
t2a-anita manifest \
  --root /data/shasegawa/t2a/datasets/anita \
  --clips-root /data/shasegawa/t2a/datasets/anita/clips \
  --output /data/shasegawa/t2a/manifests/anita_raw.jsonl

t2a-anita render-clips \
  --root /data/shasegawa/t2a/datasets/anita \
  --clips-root /data/shasegawa/t2a/datasets/anita/clips \
  --output-manifest /data/shasegawa/t2a/manifests/anita_clipped.jsonl
```

Use Anita at low weight for style/intermediate-animation regularization. Keep AnimeShooter as the primary captioned video dataset.

## Dataset manifest

Training expects JSONL records:

```json
{
  "video_path": "/data/shasegawa/t2a/datasets/animeshooter/clips/video_id/seg000_shot000.mp4",
  "caption": "In an empty classroom at sunset, a nervous schoolgirl lowers her eyes and quietly speaks as the camera slowly moves closer.",
  "source_id": "show_or_source_video_id",
  "content_type": "talking_facial_acting",
  "motion_amplitude": 2
}
```

You can also provide structured fields instead of `caption`; `text_to_anime.captioning.build_caption()` will assemble a caption from fields such as `subject`, `appearance`, `initial_state`, `action`, `secondary_motion`, `camera`, `background`, and `style`.

Validate and split by source:

```bash
t2a-build-manifest validate --manifest /data/shasegawa/t2a/manifests/animeshooter_clipped.jsonl
t2a-build-manifest split \
  --manifest /data/shasegawa/t2a/manifests/animeshooter_vn_motion_30k.jsonl \
  --train-out /data/shasegawa/t2a/manifests/train.jsonl \
  --val-out /data/shasegawa/t2a/manifests/val.jsonl \
  --val-ratio 0.02
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[video]"
export HF_HOME=/data/shasegawa/t2a/hf-cache
accelerate config
```

For two L40S GPUs, use bf16 in `accelerate config`.

## Train

Start with the small 10K experiment before scaling:

```bash
accelerate launch --num_processes 2 -m text_to_anime.train_wan_lora \
  --config configs/anisora_v32_animeshooter_360p_lora.yaml \
  --manifest /data/shasegawa/t2a/manifests/train.jsonl \
  --validation-manifest /data/shasegawa/t2a/manifests/val.jsonl
```

For this requested start, use `configs/anisora_v32_animeshooter_360p_lora.yaml`. It expects an AniSora V3.2 `WanPipeline`-compatible Diffusers folder at `/data/shasegawa/t2a/models/Index-anisora-diffusers/V3.2`; the native downloaded V3.2 folder remains at `/data/shasegawa/t2a/models/Index-anisora/V3.2`.



## Anita I2V LoRA Training

This is the current image-to-video target path. AnitaDataset is used as image-sequence video data; the first frame is the conditioning image and the sampled 3-5 second sequence is the target video.

The prepared manifests are:

```text
/data/shasegawa/t2a/manifests/anita_i2v_train.jsonl   # 317 sequences
/data/shasegawa/t2a/manifests/anita_i2v_val50.jsonl   # 50 fixed first-frame/prompt pairs
```

Train the high-noise I2V LoRA at 360p:

```bash
accelerate launch --num_processes 2 -m text_to_anime.train_wan_i2v_lora \
  --config configs/wan22_i2v_anita_360p_lora.yaml \
  --manifest /data/shasegawa/t2a/manifests/anita_i2v_train.jsonl \
  --validation-manifest /data/shasegawa/t2a/manifests/anita_i2v_val50.jsonl
```

The trainer enforces the requested constraints:

- 360p default resolution, `640x360`.
- 49 frames at 12 fps, about 4.1 seconds.
- Image-to-video conditioning from frame 0.
- VAE, text encoder, and image encoder if present are frozen.
- Gradient checkpointing is enabled.
- LoRA targets attention projections plus Wan block FFN projections for temporal dynamics: `to_q,to_k,to_v,to_out.0,ffn.net.0.proj,ffn.net.2`.
- Validation uses 50 fixed first-frame/prompt pairs by default.

Wan2.2 has high-noise and low-noise denoisers. Start with `train_stage: high`; if the high-stage adapter is useful, run a second LoRA with `--train-stage low` into a separate output directory.

## Anita Production-Stage Wan2.2 LoRA

For the production-pipeline experiment, the main all-in-one model is now a Wan2.2 I2V LoRA. The trainer keeps Wan2.2's VAE, text encoder, and image/video transformer backbone, and trains LoRA adapters on the Wan transformer. The three production tasks are expressed through the prompt plus Wan's latent video condition:

- `line_art`: optional first-frame/reference condition to a line-art shot.
- `character_color`: required line-art source shot to a character-color shot.
- `compose_refine`: required character-color source shot to the final composited shot.

Build an explicit production-shot manifest from the official Anita layout:

```bash
t2a-anita-production manifest \
  --root /data/shasegawa/t2a/datasets/anita \
  --output /data/shasegawa/t2a/manifests/anita_production.jsonl \
  --max-frames-per-shot 49 \
  --min-frames-per-shot 8 \
  --shot-split-strategy semantic \
  --semantic-split-search-radius 3
```

The manifest builder treats each Anita scene folder as the source scene and splits long scenes into contiguous semantic sub-shots before captioning. The default splitter scores adjacent-frame visual dynamics from downsampled structure and color features, then chooses high-change cut points that still respect the frame budget. If a sequence has no meaningful visual-change signal, it falls back to balanced frame-count chunks; pass `--shot-split-strategy frame_count` to force the old behavior.

Caption the sub-shots first with OpenAI vision if you want stronger text conditioning. This uses the Responses API and reads `OPENAI_API_KEY` from the environment:

```bash
t2a-caption-anita-shots \
  --input /data/shasegawa/t2a/manifests/anita_production.jsonl \
  --output /data/shasegawa/t2a/manifests/anita_production_captioned.jsonl \
  --provider openai \
  --model gpt-5.6 \
  --num-frames 3 \
  --image-detail low
```

For a local Hugging Face VLM instead, pass `--provider local --model-path /path/to/local-vlm`.

Train the Wan2.2 all-in-one production LoRA:

```bash
t2a-train-wan22-anita-production \
  --config configs/wan22_anita_production_360p_lora.yaml \
  --data /data/shasegawa/t2a/manifests/anita_production_captioned.jsonl
```

This trainer calls the Wan2.2 transformer directly and supports source-shot latent conditioning for stage 2 and stage 3 training. The stock `WanImageToVideoPipeline` inference path still only exposes first-frame/last-frame image conditioning, so use `t2a-render-wan22-anita-production` for source-shot conditioned inference.

Render with the matching sampler. It reuses the trainer's prompts and conditioning, runs both Wan2.2 experts, and writes `condition | generated | target` comparison videos plus copy-detection metrics (`psnr_gen_vs_condition` against `psnr_condition_vs_target`):

```bash
t2a-render-wan22-anita-production \
  --data /data/shasegawa/t2a/manifests/anita_production_captioned.jsonl \
  --output-dir /data/shasegawa/t2a/outputs/renders/overfit3 \
  --high-lora /data/shasegawa/t2a/outputs/wan22-anita-production-overfit3-288x512-high \
  --low-lora /data/shasegawa/t2a/outputs/wan22-anita-production-overfit3-288x512-low \
  --scenes 119_a_part000,221_a_part000,204_a_part000 \
  --high-device cuda:2 --low-device cuda:3
```

Add `--chain` to feed each stage the previous generated stage instead of the ground-truth source shot. Defaults are 40 steps, CFG 3.5, and the checkpoint scheduler's flow shift (the same one training uses).

Overfit sanity check before a long run: train each expert on 3 shots, one GPU each, then render those shots with both LoRAs:

```bash
CUDA_VISIBLE_DEVICES=2 NCCL_P2P_DISABLE=1 t2a-train-wan22-anita-production \
  --config configs/wan22_anita_production_288p_overfit3_high.yaml
CUDA_VISIBLE_DEVICES=3 NCCL_P2P_DISABLE=1 t2a-train-wan22-anita-production \
  --config configs/wan22_anita_production_288p_overfit3_low.yaml
```

On this host, direct GPU-to-GPU copies silently return zeros even though CUDA reports peer access. The renderer routes cross-GPU tensors through host memory. Set `NCCL_P2P_DISABLE=1` for any multi-GPU training.

Anita does not provide rich semantic captions, so text control is mostly task/style control unless you add stronger captions or external licensed text-video data.

## Anita Production Stages with Wan VACE

This path poses each production stage the way Wan VACE was pretrained: a control video plus a mask (1 = generate) and optional reference images. The LoRA then adapts an existing skill instead of learning a new conditioning scheme.

| Task | Control video | Mask | Reference | Target |
|---|---|---|---|---|
| `inbetween` | line-art keyframes every `keyframe_stride` frames, grey elsewhere | 0 on keys | – | line-art sequence |
| `character_color` | line-art video | 1 | one colored drawing from outside the clip | character colors |
| `compose_refine` | character layer over the recovered background | 1 | clean plate (static shots) | final composition |

Anita frame numbers are timing-sheet positions: drawings on 2s/3s skip numbers. Clips are rebuilt on the 24 fps timeline, holding each drawing, and sampled every `timeline_stride` frames (default 2, which gives 12 fps). Short shots use the longest valid Wan length (4k+1) instead of being stretched.

Recover background plates first. Static shots get a clean median plate; moving shots get per-frame backgrounds:

```bash
t2a-anita-bg-plates   # writes /data/shasegawa/t2a/manifests/anita_background_plates.jsonl
```

Download VACE 1.3B for fast iteration, then train. Whole shots are held out for validation (`split.json`, `validation_log.jsonl` with fixed-noise, fixed-timestep losses):

```bash
scripts/download_wan21_vace.sh 1.3B
CUDA_VISIBLE_DEVICES=0,2,3 NCCL_P2P_DISABLE=1 accelerate launch --num_processes 3 --mixed_precision bf16 \
  -m text_to_anime.train_wan_vace_anita --config configs/wan21_vace_1.3b_anita_480p_lora.yaml
```

Render held-out shots with the stock `WanVACEPipeline`. The trainer builds its conditioning with the pipeline's own `prepare_video_latents`/`prepare_masks`, so inference sees exactly the training inputs:

```bash
t2a-render-wan-vace-anita --config configs/wan21_vace_1.3b_anita_480p_lora.yaml \
  --lora /data/shasegawa/t2a/outputs/wan21-vace-1.3b-anita-480p-r1 \
  --output-dir /data/shasegawa/t2a/outputs/renders/vace-r1-val --device cuda:0
```

Omit `--lora` to render the base VACE model as a baseline.

## Image-To-Video

Image-to-video uses the supplied image as frame 0. For dataset workflows, extract the first frame from each training clip into `first_frame_path`:

```bash
t2a-extract-frames \
  --manifest /data/shasegawa/t2a/manifests/animeshooter_clipped.jsonl \
  --output-root /data/shasegawa/t2a/datasets/conditioning_frames/animeshooter \
  --output-manifest /data/shasegawa/t2a/manifests/animeshooter_i2v.jsonl \
  --frame-index 0
```

Download the Wan2.2 I2V Diffusers checkpoint for image-conditioned inference:

```bash
scripts/download_wan22_i2v.sh
```

Run image-to-video inference:

```bash
t2a-infer-wan-i2v \
  --pretrained-model-name-or-path /data/shasegawa/t2a/models/Wan2.2-I2V-A14B-Diffusers \
  --image /data/shasegawa/t2a/datasets/conditioning_frames/animeshooter/example_first_frame.jpg \
  --prompt "A schoolgirl in a classroom quietly begins speaking, with subtle blinking and gentle hair movement. Slow push-in from medium shot to close-up." \
  --output /data/shasegawa/t2a/outputs/samples/i2v_sample.mp4 \
  --height 360 \
  --width 640 \
  --num-frames 49 \
  --fps 12
```

For first-last-frame models, pass `--last-image`; otherwise only the initial image is used.

## Infer

```bash
t2a-infer-wan \
  --pretrained-model-name-or-path /data/shasegawa/t2a/models/Index-anisora-diffusers/V3.2 \
  --lora-path /data/shasegawa/t2a/outputs/anisora-v32-animeshooter-vn-lora \
  --prompt "A high-school girl in a navy uniform stands beside a classroom window at sunset. She lowers her eyes and quietly begins speaking while her hair moves slightly in the breeze. Slow push-in from medium shot to close-up." \
  --output /data/shasegawa/t2a/outputs/samples/sample.mp4 \
  --height 360 \
  --width 640 \
  --num-frames 49 \
  --fps 12
```

## Notes

- Do not deduplicate repeated frames inside a clip. Held frames are part of anime timing.
- Split train and validation by title or source video, not by random clip.
- Remove credits, subtitles, watermarks, split screens, and severe compression artifacts before training.
- Keep commercial licensing separate from the research pipeline. web-scale anime datasets are usually not suitable for commercial deployment without rights clearance.

