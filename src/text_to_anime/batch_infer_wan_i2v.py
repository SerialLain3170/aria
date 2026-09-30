from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .infer_wan import DEFAULT_NEGATIVE_PROMPT, dtype_from_name, prompt_for_anime
from .infer_wan_i2v import load_rgb, maybe_pipeline_multiple, resize_crop, rounded_dimensions_for_image
from .manifest import read_jsonl
from .video import assert_wan_frame_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch raw Wan image-to-video inference.")
    parser.add_argument("--manifest", required=True, help="JSONL with image/first_frame/first_frame_path and caption/prompt.")
    parser.add_argument("--pretrained-model-name-or-path", default="/data/shasegawa/t2a/models/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--revision")
    parser.add_argument("--variant")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max-area", type=int, default=640 * 360)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--guidance-scale-2", type=float)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--no-append-anime-clauses", action="store_true")
    return parser


def record_image_path(record: dict) -> str:
    for key in ("image", "image_path", "first_frame_path", "first_frame"):
        value = record.get(key)
        if value:
            return str(value)
    raise ValueError(f"record has no image path fields: {record}")


def record_prompt(record: dict) -> str:
    return str(record.get("prompt") or record.get("caption") or "A hand-drawn animation shot with subtle motion.")


def main() -> None:
    args = build_parser().parse_args()
    assert_wan_frame_count(args.num_frames)

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    records = read_jsonl(args.manifest)[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = dtype_from_name(args.dtype)
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float32,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    multiple = maybe_pipeline_multiple(pipe)
    generator = torch.Generator(device=pipe._execution_device).manual_seed(args.seed)
    written = []
    metadata = []
    for index, record in enumerate(records):
        image_path = record_image_path(record)
        image = load_rgb(image_path)
        if args.height and args.width:
            height = args.height // multiple * multiple
            width = args.width // multiple * multiple
        else:
            height, width = rounded_dimensions_for_image(image, max_area=args.max_area, multiple=multiple)
        image = resize_crop(image, height=height, width=width)
        prompt = prompt_for_anime(
            record_prompt(record),
            aesthetic_score=5.5,
            motion_score=2.0,
            append=not args.no_append_anime_clauses,
        )
        call_kwargs = {
            "image": image,
            "prompt": prompt,
            "negative_prompt": args.negative_prompt or None,
            "height": height,
            "width": width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "generator": generator,
        }
        if args.guidance_scale_2 is not None:
            call_kwargs["guidance_scale_2"] = args.guidance_scale_2
        frames = pipe(**call_kwargs).frames[0]
        name = str(record.get("clip_id") or f"sample_{index:03d}").replace("/", "_")
        output_path = output_dir / f"{index:03d}_{name}.mp4"
        export_to_video(frames, str(output_path), fps=args.fps)
        written.append(str(output_path))
        metadata.append({"output": str(output_path), "image": image_path, "prompt": prompt})
        print(f"wrote {output_path}")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"videos": written}, indent=2))


if __name__ == "__main__":
    main()
