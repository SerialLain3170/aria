# Future Direction: Editable Three-Phase Anime Generation

## North star

The project should become a prompt-guided anime shot generator that produces editable
production artifacts, not only a final MP4.

The target user flow is:

```text
shot prompt + optional character/scene references + optional key drawings
                                  |
                                  v
                    Phase 1: motion and line art
                                  |
                                  v
                    Phase 2: character color
                                  |
                                  v
                    Phase 3: final composition
                                  |
                                  v
        line-art video + color video + final video + reproducibility record
```

Each phase should be independently viewable, editable, rerunnable, and replaceable. Text
defines intent; intermediate videos and references preserve timing, character identity,
and scene continuity.

This direction is deliberately different from treating anime generation as a single
opaque prompt-to-video call. The final experience may begin with only a prompt, but the
system should expose the same intermediate representations that an artist would use to
correct the result.

## Current baseline

The current Wan VACE LoRA demonstrates that one adapter can learn three related
production-stage transformations:

1. sparse line-art keys to a complete line-art sequence;
2. line art plus a color reference to a character-colored sequence;
3. a character layer plus recovered background context to a final composite.

The baseline has several important constraints:

- Phase 1 is primarily in-betweening, not unrestricted prompt-to-line-art generation.
- Phase 2 is materially more reliable with a color reference.
- Phase 3 is materially more reliable with a background or plate reference.
- AnitaDataset teaches stage transformation well but provides limited open-domain semantic
  variety.
- validation is strongest phase by phase; chained error is not yet a first-class training
  signal.
- the research workflow produces artifacts, but there is no single shot-level generation
  command or durable run manifest.

The roadmap should improve these constraints without losing the pipeline structure that
makes the current system controllable.

## Product principles

### Preserve intermediate ownership

Every generation should save Phase 1, Phase 2, and Phase 3 outputs. A user must be able to
replace a generated intermediate with an edited one and resume downstream generation.

### Prefer explicit control over hidden state

Character sheets, palette references, key drawings, camera instructions, and background
plates should be explicit inputs with provenance. They should not exist only as embeddings
inside an unrecoverable session.

### Make the easy path coherent

One command should orchestrate all phases with sensible defaults. Advanced users should
still be able to run a single phase, change its prompt or reference, and regenerate only
downstream outputs.

### Treat failure localization as a feature

The system should identify whether a failure came from motion, drawing, color, background,
or chaining. A pipeline is valuable only if it makes correction cheaper than regenerating
the entire shot.

### Keep evaluation paired with generation

Every run should emit visual comparisons, phase metrics, exact prompts, seeds, model and
adapter identities, and input checksums.

## Proposed generation contract

A future shot specification should be a portable YAML or JSON file:

```yaml
shot_id: opening_closeup_001
prompt: >
  A determined young pilot looks up as warning lights sweep across the cockpit.
  Tight close-up, subtle breathing and blinking, slow push-in, tense blue-red lighting.
duration_seconds: 4
fps: 12
width: 832
height: 480
seed: 4096

character_reference: references/pilot_turnaround.png
color_reference: references/pilot_palette.png
background_reference: references/cockpit_plate.png
key_drawings:
  - frame: 0
    path: keys/0000.png
  - frame: 24
    path: keys/0024.png
  - frame: 48
    path: keys/0048.png

phases:
  line_art:
    prompt_suffix: clean production line art, stable facial proportions
  character_color:
    prompt_suffix: flat cel colors, preserve every line
  final_composite:
    prompt_suffix: warning-light sweep, subtle bloom, finished anime composite
```

The orchestrator should create a shot directory such as:

```text
opening_closeup_001/
├── shot.yaml
├── run.json
├── inputs/
├── phase1_line_art/
│   ├── generated.mp4
│   ├── control.mp4
│   └── metrics.json
├── phase2_character_color/
│   ├── generated.mp4
│   ├── control.mp4
│   └── metrics.json
├── phase3_final_composite/
│   ├── generated.mp4
│   ├── control.mp4
│   └── metrics.json
└── final.mp4
```

This format becomes the boundary between a UI, batch production, evaluation, and model
research.

## Roadmap

### Milestone 0: Reproducible baseline

Before changing model scale or task design, make the existing result reproducible.

Deliverables:

- one preflight command for model paths, manifests, `ffmpeg`, CUDA, disk, and GPU topology;
- code revision, package versions, config, manifest hashes, and base-model identity saved
  with every run;
- full Accelerate checkpoint/resume, including optimizer, scheduler, sampler, and step;
- a fixed golden validation set spanning all three phases;
- automatic base-versus-LoRA and teacher-forced-versus-chained reports;
- CI for CPU-only data, conditioning, masking, splitting, and metric tests.

Exit criteria:

- a new operator can reproduce the checked-in validation summary from documented assets;
- an interrupted job resumes without changing its sample sequence or learning-rate state;
- no validation shot, parent scene, or related phase record leaks into training.

### Milestone 1: Shot-level orchestrator

Build a single CLI around the proposed shot specification.

Proposed interface:

```bash
t2a-generate-shot --shot shot.yaml --output outputs/opening_closeup_001
```

Required behaviors:

- run all three phases or a requested subset;
- resume from an edited intermediate;
- keep seeds stable per phase;
- validate dimensions, frame count, references, and masks before allocating the model;
- load the base checkpoint and LoRA once when possible;
- generate a contact sheet and synchronized three-panel MP4 automatically;
- mark every phase as reference-conditioned, prompt-only, or artist-supplied.

Exit criteria:

- changing a Phase 2 prompt reruns only phases 2 and 3;
- replacing Phase 1 with an artist edit requires no code changes;
- the output directory alone is sufficient to reconstruct the run.

### Milestone 2: Stronger Phase 1 motion planning

Phase 1 is the largest gap between the current checkpoint and prompt-first generation.
The next version should support a spectrum of control:

1. dense keys for faithful in-betweening;
2. sparse start/middle/end keys for assisted animation;
3. first-frame plus pose or trajectory controls;
4. prompt-only motion blocking as an exploratory mode.

Research directions:

- curriculum training that gradually removes intermediate keys;
- explicit pose, depth, edge, or optical-flow controls;
- camera-path tokens separated from subject-motion tokens;
- keyframe-confidence masks rather than binary known/unknown masks;
- motion-amplitude conditioning and held-frame/exposure prediction;
- a lightweight planning model that proposes keys before VACE in-betweening.

Phase 1 should continue to preserve held frames. Frame deduplication would erase timing
decisions and should not be used as a preprocessing shortcut.

Exit criteria:

- supplied keys remain pixel- and identity-stable;
- motion between keys is smooth without collapsing anime holds;
- prompt motion and camera instructions are measurably reflected in the result;
- reducing key density degrades gracefully rather than changing the character or scene.

### Milestone 3: Character identity and palette memory

Phase 2 should evolve from one-shot color reference conditioning into persistent character
control.

Deliverables:

- a structured character package containing turnaround images, palette swatches, clothing
  variants, and textual attributes;
- reference selection based on pose/view similarity instead of a random outside frame;
- palette and region consistency losses;
- face/hair/clothing identity metrics over time;
- support for multiple named characters without color leakage;
- optional per-character LoRA or reference adapter when the shared model is insufficient.

The system should prefer a reference that is close enough to guide the current view but
outside the target window so it cannot simply copy the answer.

Exit criteria:

- palette drift stays below a defined threshold over the shot;
- identity remains stable across front, profile, and partial occlusion;
- changing the palette reference changes color without changing motion or line topology;
- two-character shots preserve distinct palettes.

### Milestone 4: Scene, camera, and compositing control

Phase 3 should separate background generation from final lighting/compositing instead of
forcing one transformation to invent both implicitly.

Potential sub-pipeline:

```text
scene prompt/reference -> background plate or moving background
character color + background -> integration
integration + lighting/effect prompt -> final composite
```

Research directions:

- camera-aware background generation using depth or homography controls;
- explicit alpha/matte prediction for character integration;
- static versus moving-camera routing;
- light-direction and color-script conditioning;
- effect layers for bloom, smoke, rain, speed lines, and particles;
- temporal relighting losses that avoid flicker;
- background reuse across a sequence of shots.

Exit criteria:

- the character remains registered to the background and ground plane;
- camera movement is consistent across character and scene;
- relighting does not erase line work or alter identity;
- static plates remain stable while moving backgrounds remain temporally coherent.

### Milestone 5: Chained and joint training

The current phases are validated mainly with ground-truth controls. Production inference
must tolerate generated upstream inputs.

Recommended progression:

1. keep phase-specific teacher-forced batches for stable learning;
2. mix in cached generated upstream controls;
3. add controlled corruption that resembles measured upstream errors;
4. fine-tune on short chained sequences;
5. compare one shared adapter with phase-specific adapters and routed mixtures.

Do not begin with fully end-to-end training. It would make attribution difficult and could
allow downstream phases to hide upstream failures.

Exit criteria:

- chained validation quality approaches teacher-forced quality;
- Phase 2 remains robust to realistic Phase 1 line variation;
- Phase 3 remains robust to realistic Phase 2 palette and matte errors;
- improvement in final video does not coincide with worse editable intermediates.

### Milestone 6: Broader, rights-cleared data

Anita alone cannot provide broad text-to-video semantics, varied cinematography, or enough
character diversity.

The data strategy should add rights-cleared sources with explicit roles:

- production-stage pairs for transformation supervision;
- captioned finished animation for semantic and camera understanding;
- character sheets and palettes for identity conditioning;
- background plates and layout art for scene generation;
- timing sheets or exposure metadata for anime-specific motion;
- licensed effects layers for compositing.

Every record should carry provenance, license, source grouping, transformation history,
and allowed-use metadata. Dataset growth should not weaken whole-title or whole-scene split
boundaries.

Exit criteria:

- every training sample has machine-readable provenance and allowed use;
- semantic diversity improves without degrading the three-phase transformations;
- validation includes unseen characters, scenes, camera patterns, and motion categories.

## Model strategy

### Scale only after task validation

The 1.3B VACE model is appropriate for fast task-design iteration. A larger VACE backbone
should be evaluated only after conditioning, splits, and metrics are stable. Otherwise
scale will make an ambiguous experiment slower rather than more informative.

### Compare shared and phase-specific adapters

One shared LoRA encourages transfer and simplifies deployment, but the three phases have
different output statistics. Compare:

- one shared balanced adapter;
- one adapter per phase;
- a shared trunk with small phase-specific adapters;
- routed or weighted adapter composition.

The comparison should hold the base model, data, prompts, seeds, and parameter budget
constant.

### Keep pipeline-native conditioning

Use the base model's documented control/mask/reference mechanisms whenever possible.
Inventing an unmatched latent layout creates a training/inference gap and asks a small
dataset to teach a new interface as well as a new visual task.

### Separate planning from rendering when useful

A compact motion/layout planner may be more efficient than asking a large renderer to
infer timing, pose, camera, and appearance jointly. Planning outputs should remain visible
and editable.

## Evaluation framework

Evaluation should report results per phase and end to end.

### Phase 1

- keyframe preservation;
- line topology and identity stability;
- temporal smoothness between keys;
- held-frame timing accuracy;
- prompt action and camera adherence.

### Phase 2

- palette error against reference;
- line preservation;
- character identity consistency;
- background cleanliness;
- temporal color flicker.

### Phase 3

- matte/edge integration;
- background and camera coherence;
- lighting consistency;
- saturation and exposure bounds;
- temporal flicker and motion preservation.

### Chained system

- teacher-forced versus chained quality gap;
- per-phase retry rate;
- final-shot human preference;
- edit propagation correctness;
- time and GPU cost per accepted shot.

PSNR, saturation, and mean motion remain useful diagnostics but are not acceptance metrics
by themselves. A fixed human review rubric should score motion, identity, line quality,
palette, composition, prompt adherence, and editability.

## Recommended next experiment

The most informative next experiment is a sparse-key Phase 1 curriculum followed by
chained evaluation:

1. choose a small leakage-safe set of shots that contain all three stages;
2. train Phase 1 with the current key density as the control group;
3. train variants with progressively fewer keys while keeping frame windows fixed;
4. feed each Phase 1 result through the existing reference-conditioned phases 2 and 3;
5. measure key preservation, motion quality, identity, chained degradation, and human
   preference;
6. select the sparsest key regime that does not materially reduce final-shot acceptance.

This experiment directly advances prompt-first usability while retaining the strongest
part of the current approach: explicit, editable production phases.

## Decisions to avoid prematurely

- Do not remove intermediate outputs in favor of an end-to-end-only model.
- Do not treat prompt-only extrapolation as equivalent to reference-conditioned quality.
- Do not scale to a much larger backbone before the validation protocol is stable.
- Do not mix generated and ground-truth controls without recording which one was used.
- Do not optimize only final-frame metrics at the expense of motion or editability.
- Do not expand training data without source-group splits and rights metadata.

## Success definition

The project succeeds when a user can describe a short anime shot, provide as much or as
little production control as they have, and receive:

- a coherent line-art animation;
- stable character color;
- a finished composite;
- editable intermediate artifacts;
- a reproducible record of every input and model decision;
- clear feedback about which phase should be corrected when the result is not acceptable.

The goal is not merely to generate a video. It is to make anime generation behave like an
inspectable, controllable production pipeline.
