# Project documentation

These documents describe the repository as inspected on 2026-09-30. The code is the
source of truth; paths under `/data/shasegawa/t2a` are external runtime state and are not
part of this repository.

## Start here

- [Architecture](architecture.md) — components, data flows, model paths, module ownership,
  and artifact boundaries.
- [Training methodology](training-methodology.md) — dataset construction, conditioning,
  flow-matching objectives, LoRA strategy, validation, and experiment protocol.
- [Future direction](future-direction.md) — product north star, research roadmap,
  engineering milestones, evaluation gates, and the recommended next experiment.
- [Takeover guide](takeover/README.md) — what a new maintainer needs to know first,
  current limitations, risks, and suggested priorities.
- [Operations runbook](takeover/runbook.md) — environment setup, preflight checks, data
  preparation, training, rendering, and failure recovery.
- [Current-state inventory](takeover/current-state.md) — snapshot-specific status and
  what could and could not be verified.

## Scope

The repository is a research and production-prototyping toolkit for adapting Diffusers-
compatible Wan/AniSora video models to anime. It does not pretrain a foundation model.
Its principal mechanism is LoRA adaptation of frozen video diffusion transformers.

There are four implemented model families:

1. Text-to-video LoRA on Wan/AniSora.
2. First-frame image-to-video LoRA on Wan2.2 I2V.
3. Source-shot-conditioned Wan2.2 production-stage LoRA.
4. VACE-native production-stage LoRA, which most closely matches the intended Anita
   in-betweening, color, and compositing workflow.

An experimental standalone 3D U-Net is also retained. It is useful as a supervised
baseline, but it is separate from the pretrained Wan inference stack.
