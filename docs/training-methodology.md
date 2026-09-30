# Training methodology

## Objectives and experiment philosophy

The project adapts pretrained video generators instead of attempting full pretraining.
Experiments should progress from a tiny overfit test, to a deterministic held-out test, to
a full run. A lower training loss alone is not acceptance evidence: a production-stage
model can minimize loss by copying its condition, losing color, or suppressing motion.

The primary production hypothesis is that Anita's stages should be represented through
the conditioning mechanism the base model already learned. For that reason the VACE path
is the preferred production experiment. The Wan2.2 source-shot path remains useful for
comparing the A14B two-expert backbone, and the standalone U-Net is a supervised baseline.

## Data methodology

### AnimeShooter/Sakuga T2V

1. Convert source metadata into the shared JSONL schema.
2. Extract physical clips with exact shot boundaries.
3. Preserve native descriptive captions when present.
4. Score records by resolution, 2–6.5 second duration preference, aesthetic score,
   moderate motion, caption richness, camera language, and text/artifact penalties.
5. Limit clips per source to reduce title/episode domination.
6. Build the default VN-oriented mixture:
   30% talking/facial acting, 20% subtle idle motion, 15% walking/turning/entering,
   15% emotional gestures, 10% camera/composition, and 10% dynamic action.
7. Split deterministically by `source_id` (or the next available source key), not by clip.

The hash-based source split is stable as the manifest is reordered. Adding new sources
does not reshuffle existing ones.

### Anita production data

Frames are paired by stem across sketch, color, and composition stages. Production tasks
are derived only where required pairs exist. Long sequences are divided into contiguous
sub-shots; semantic cut selection is preferred, with balanced frame-count fallback.

Captioning selects up to three centered representative frames, preferring the final
composition, then color, then line art. One semantic caption is shared across all task
records for that sub-shot, while each task receives its own imperative prefix. Caption
outputs retain provider, model, and source-frame provenance.

For VACE, numeric drawing IDs are interpreted as 24 fps timing positions. Missing numbers
become held frames, and sampling every second timeline frame produces 12 fps without
destroying exposure timing. The longest valid Wan length not exceeding `max_frames` is
used; valid lengths are `4k+1`.

### Background recovery

The character-color layer's alpha separates character from background in the final
composition. The plate builder dilates this mask, computes a median from visible
background pixels, fills holes through a push-pull pyramid, and classifies a shot as static
from median background residual. Static shots share a clean plate; moving shots receive
per-frame interpolated backgrounds. The plate index records residual, coverage, size, and
static/moving classification for audit.

## Shared preprocessing

Training frames are cover-resized, then randomly cropped; validation uses centered crops.
RGB is normalized to `[-1, 1]`. Generic video training samples a contiguous temporal
window. A short source sequence is resampled across its entire extent, which repeats frame
indices where necessary.

The VAE stays frozen. By default Wan latent posterior mode/mean is used rather than a
sample, reducing target variance. Wan latent channels are normalized using the checkpoint's
`latents_mean` and `latents_std`. Text is truncated/padded to 512 tokens by default and the
text encoder is frozen.

## Flow-matching objective

The Wan trainers share the same rectified-flow target. Let `x` be a normalized clean video
latent, `e ~ N(0, I)` be Gaussian noise, and `s` be the scheduler sigma for a sampled
timestep. The noisy input and velocity target are:

```text
x_s = (1 - s) x + s e
v*  = e - x
L   = mean((v_theta(x_s, conditions, t) - v*)^2)
```

Timesteps are sampled uniformly from scheduler entries. For Wan2.2, `boundary_ratio`
divides the schedule: a `high` run samples at or above the boundary and adapts
`transformer`; a `low` run samples below it and adapts `transformer_2`. VACE 1.3B has one
transformer and samples the full schedule.

## LoRA strategy

All pretrained VAE, text encoder, image encoder (when present), and transformer base
weights are frozen. PEFT adapters are initialized with Gaussian LoRA weights.

The generic T2V config targets attention projections:

```text
to_q, to_k, to_v, to_out.0
```

I2V and production configs also target feed-forward projections to increase capacity for
temporal and transformation behavior:

```text
to_q, to_k, to_v, to_out.0, ffn.net.0.proj, ffn.net.2
```

Typical rank/alpha is 16/16 for initial runs, 32/32 for the balanced Wan2.2 and VACE
production experiments. Only adapter state is saved for LoRA runs, along with resolved
arguments and descriptive metadata.

## Conditioning methods

### Text-to-video

The transformer receives noisy target latents and text embeddings only. Captions include
semantic content, optional quality/motion scores, and a no-text clause.

### First-frame I2V

Frame zero is placed into an otherwise zero video and encoded by the VAE. A mask marks the
known first frame, adjusted for the VAE's temporal compression. Mask channels and condition
latents are concatenated to the noisy target before the transformer.

### Wan2.2 production source-shot conditioning

`character_color` and `compose_refine` use the aligned full previous-stage video with a
full mask. `line_art` has no natural previous stage, so it probabilistically uses target
frame zero, a still reference, or no condition. Condition and text dropout are available
for robustness. The balanced configs set condition dropout and text dropout to zero and
strongly prefer a first-frame condition for line art to avoid accidentally turning the
task into text-only generation.

Task prompts contain explicit tokens/instructions and are followed by scene prompt and
caption. Since Anita's task counts are imbalanced, `WeightedRandomSampler` assigns each
record weight `desired_task_weight / records_in_task`; this first equalizes task probability
and then applies the requested task mixture.

### VACE-native production conditioning

- In-betweening keeps sparse line-art frames with mask 0 and asks VACE to generate masked
  positions from gray placeholders.
- Character color uses line art as control, mask 1 over the sequence, and a colored drawing
  from outside the sampled window as a reference.
- Compose/refine composites the RGBA character layer over a recovered per-frame background,
  uses mask 1, and adds a clean plate reference for static shots.

The training code calls the pipeline's own conditioning methods, then feeds
`control_hidden_states` to the transformer. If references exist, their latent targets are
prepended in the same way as stock VACE inference.

## VACE change-weighted loss

Full-mask transformations can collapse into input copying because much of each frame is
already correct. VACE therefore computes a spatial-temporal change map where target and
control differ by more than `change_threshold` in normalized RGB. It max-pools the map to
latent resolution and over each VAE temporal group, dilates it spatially, and assigns:

```text
weight = loss_floor + (1 - loss_floor) * changed
```

Default `loss_floor=0.2` retains a smaller loss on unchanged areas while changed regions
receive full weight. The final MSE is normalized by total spatial weights and latent
channels. Reference latent positions receive floor weight.

## Standalone 3D U-Net objective

The experimental model directly predicts RGB frames. Its base loss is foreground-weighted
L1. Color and compose tasks upweight mask regions; a temporal L1 term matches target frame
deltas; line-art samples add a Laplacian edge loss. This is supervised regression, not
flow matching, and its checkpoints include model and optimizer state.

## Optimization defaults

Representative checked-in configurations are:

| Path | Resolution / frames | Steps | LoRA | Effective-batch controls |
|---|---|---:|---|---|
| AniSora T2V | 640×360 / 49 @ 12 fps | 12,000 | r16/a16 | batch 1, accumulation 8 |
| Wan2.2 Anita I2V | 640×360 / 49 @ 12 fps | 3,000 | r16/a16 | batch 1, accumulation 8 |
| Wan2.2 production balanced | 512×288 / 33 @ 8 fps | 4,000 per expert | r32/a32, dropout .05 | batch 1, accumulation 12 |
| VACE 1.3B production | 832×480 / up to 33 @ 12 fps | 1,500 | r32/a32 | batch 1, accumulation 2, 3 processes |

LoRA runs use AdamW, cosine decay, warmup, gradient clipping at 1.0, bf16, and gradient
checkpointing. Effective global batch is:

```text
train_batch_size × gradient_accumulation_steps × number_of_processes
```

VAE tiling is enabled for most Wan paths but disabled in the VACE config. The VACE VAE is
bf16 by default for speed; its renderer uses the same dtype so training/inference
conditioning stays aligned.

## Validation methodology

### Generic T2V

Every configured interval, render up to four fixed prompts (or validation-manifest
captions) with the current adapter, eight steps, guidance 1.0, and seed `seed + step`.

### First-frame I2V

The CLI requires 50–100 fixed validation pairs. It renders the deterministic centered
sequence's first frame with the record caption, eight steps, guidance 5.0, and seed
`seed + step`.

### Wan2.2 production

There is no validation loop inside this trainer. The required protocol is external:

1. Overfit three known scenes independently for high and low experts.
2. Render both adapters together on those same scenes.
3. Check that color does not return unchanged line art and compose does not copy the
   character-color input.
4. Run teacher-forced held-out renders.
5. Run chained renders only after every individual stage works.

The renderer records PSNR, saturation, and motion. A high generated-vs-condition PSNR,
especially when generated-vs-target does not improve over condition-vs-target, is a copy
failure.

### VACE production

All tasks for a whole shot share its split. Explicit validation scenes override the stable
hash ratio. Validation uses centered windows and deterministic augmentation. At each
interval, loss is averaged across fixed noise at three scheduler regions and reported per
task plus global mean. The renderer then uses the stock VACE pipeline on the same held-out
task definitions and emits visual panels and copy-detection metrics.

## Experiment acceptance gates

Before scaling a run:

1. Manifest paths resolve and task/source counts match expectations.
2. Train/validation source or shot sets have no overlap.
3. A tiny overfit run changes the condition in the intended regions.
4. Both base-model and adapted renders exist for the same seeds/settings.
5. Held-out per-task loss improves without motion/saturation collapse.
6. Teacher-forced visual results pass before chained evaluation.
7. The final output directory contains resolved arguments, adapter metadata, split, logs,
   and render summaries sufficient to reproduce the decision.

## Known methodological limitations

- Anita has weak native semantic text, so captions mostly improve scene conditioning; the
  dataset primarily teaches stage transformations.
- Generic T2V filtering uses heuristic metadata scores rather than decoded visual QA.
- Wan2.2 production training has no integrated held-out loss or render hook.
- PSNR, saturation, and mean motion do not measure perceptual animation quality; human
  review remains necessary.
- LoRA trainers do not implement optimizer/scheduler resume. Their `checkpoint-N`
  directories are adapter snapshots, not full resumable training state.
- The standalone U-Net validation split is record-random, so related task records from one
  shot may cross splits; do not treat it as a leakage-safe production metric.
