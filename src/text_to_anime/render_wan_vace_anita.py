from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .infer_wan import DEFAULT_NEGATIVE_PROMPT
from .render_wan22_anita_production import hstack_videos, load_lora, video_metrics, write_video
from .train_wan_lora import load_config
from .train_wan_vace_anita import (
    TIMELINE_FPS,
    build_sample,
    caption_for,
    load_captions,
    load_split,
    prompt_for,
)
from .train_wan_vace_anita import (
    build_parser as build_train_parser,
)

# Stock WanVACEPipeline inference for LoRAs from train_wan_vace_anita. Conditions come from the
# trainer's build_sample, so renders use exactly the training task definitions.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Anita production stages with a Wan VACE LoRA.")
    parser.add_argument("--config", required=True, help="The training config (data, split, and resolution).")
    parser.add_argument("--lora", help="LoRA dir or .safetensors; omit to render the base VACE model.")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--scenes", help="Optional comma-separated work/scene or scene names within the split.")
    parser.add_argument("--tasks", help="Optional subset of the config's tasks.")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, help="Defaults to the checkpoint scheduler config.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--device", default="cuda:0")
    return parser


def to_pil_frames(frames: torch.Tensor) -> list[Image.Image]:
    return [Image.fromarray(frame.permute(1, 2, 0).numpy()) for frame in frames]


def mask_to_pil_frames(mask: torch.Tensor) -> list[Image.Image]:
    values = (mask[:, 0].numpy() * 255).round().astype(np.uint8)
    return [Image.fromarray(value).convert("RGB") for value in values]


def select_sources(args: argparse.Namespace, train_args: argparse.Namespace) -> list[dict[str, Any]]:
    train_sources, val_sources = load_split(train_args)
    sources = val_sources if args.split == "val" else train_sources
    if args.scenes:
        wanted = {item.strip() for item in args.scenes.split(",") if item.strip()}
        sources = [source for source in sources if source["key"] in wanted or source["scene"] in wanted]
    if args.tasks:
        wanted_tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
        sources = [source for source in sources if source["task"] in wanted_tasks]
    if not sources:
        raise ValueError(f"no {args.split} clips selected")
    return sources[: args.max_samples]


def main() -> None:
    args = build_parser().parse_args()
    train_args = build_train_parser(load_config(args.config)).parse_args([])
    sources = select_sources(args, train_args)
    captions = load_captions(train_args.captions)

    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    vae = AutoencoderKLWan.from_pretrained(
        train_args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=getattr(torch, train_args.vae_dtype),
    )
    pipe = WanVACEPipeline.from_pretrained(
        train_args.pretrained_model_name_or_path, vae=vae, torch_dtype=torch.bfloat16
    )
    if args.flow_shift is not None:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
    if args.lora:
        load_lora(pipe, pipe.transformer, args.lora, adapter_name="anita", scale=args.lora_scale)
    pipe.to(args.device)

    output_dir = Path(args.output_dir)
    fps = TIMELINE_FPS // train_args.timeline_stride
    summary = []
    for index, source in enumerate(sources):
        sample = build_sample(
            source,
            height=train_args.height,
            width=train_args.width,
            max_frames=train_args.max_frames,
            timeline_stride=train_args.timeline_stride,
            keyframe_stride=train_args.keyframe_stride,
            compose_plate_reference=train_args.compose_plate_reference,
            rng=random.Random(index),
            center=True,
        )
        prompt = prompt_for(source["task"], caption_for(captions, source["key"], sample["drawings"]))
        num_frames = sample["target"].shape[0]
        output = pipe(
            video=to_pil_frames(sample["control"]),
            mask=mask_to_pil_frames(sample["mask"]),
            reference_images=sample["reference"],
            prompt=prompt,
            negative_prompt=args.negative_prompt or None,
            height=train_args.height,
            width=train_args.width,
            num_frames=num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=args.device).manual_seed(args.seed + index),
            output_type="np",
        )
        generated = torch.from_numpy((output.frames[0] * 255).round().clip(0, 255).astype(np.uint8))
        generated = generated.permute(0, 3, 1, 2).contiguous()

        stem = output_dir / source["key"].replace("/", "__") / source["task"]
        stem.parent.mkdir(parents=True, exist_ok=True)
        panels = [sample["control"], generated, sample["target"]]
        write_video(generated, stem.with_name(f"{stem.name}_generated.mp4"), fps)
        write_video(hstack_videos(panels), stem.with_name(f"{stem.name}_control_generated_target.mp4"), fps)
        if sample["reference"] is not None:
            sample["reference"].save(stem.with_name(f"{stem.name}_reference.png"))
        # Copy detection is only meaningful for full-mask tasks; in-betweening keeps its keyframes.
        kind = "source_shot" if source["task"] != "inbetween" else "keyframes"
        info = {
            "key": source["key"],
            "task": source["task"],
            "prompt": prompt,
            "num_frames": num_frames,
            "drawings": sample["drawings"],
            "metrics": video_metrics(generated, sample["target"], sample["control"], kind),
        }
        stem.with_name(f"{stem.name}_render.json").write_text(json.dumps(info, indent=2))
        summary.append(info)
        print(f"{source['key']} {source['task']}: {json.dumps(info['metrics'])}")

    (output_dir / "render_summary.json").write_text(
        json.dumps({"settings": vars(args), "renders": summary}, indent=2)
    )
    print(f"wrote {output_dir / 'render_summary.json'}")


if __name__ == "__main__":
    main()
