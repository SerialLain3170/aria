from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .infer_wan import DEFAULT_NEGATIVE_PROMPT, dtype_from_name, prompt_for_anime
from .video import assert_wan_frame_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Wan image-to-anime-video inference from an initial frame.")
    parser.add_argument("--pretrained-model-name-or-path", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--revision")
    parser.add_argument("--variant")
    parser.add_argument("--lora-path")
    parser.add_argument("--lora-weight-name")
    parser.add_argument("--adapter-name", default="anime")
    parser.add_argument("--load-into-transformer-2", action="store_true")
    parser.add_argument("--image", required=True, help="Initial frame image. It is used as frame 0.")
    parser.add_argument("--last-image", help="Optional ending frame for first-last-frame models.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
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
    parser.add_argument("--aesthetic-score", type=float, default=5.5)
    parser.add_argument("--motion-score", type=float, default=2.0)
    parser.add_argument("--no-append-anime-clauses", action="store_true")
    return parser


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def rounded_dimensions_for_image(image: Image.Image, *, max_area: int, multiple: int) -> tuple[int, int]:
    aspect = image.height / image.width
    height = int(round(np.sqrt(max_area * aspect)))
    width = int(round(np.sqrt(max_area / aspect)))
    height = max(multiple, height // multiple * multiple)
    width = max(multiple, width // multiple * multiple)
    return height, width


def resize_crop(image: Image.Image, *, height: int, width: int) -> Image.Image:
    resize_ratio = max(width / image.width, height / image.height)
    resized = image.resize((round(image.width * resize_ratio), round(image.height * resize_ratio)), Image.Resampling.LANCZOS)
    left = max((resized.width - width) // 2, 0)
    top = max((resized.height - height) // 2, 0)
    return resized.crop((left, top, left + width, top + height))


def maybe_pipeline_multiple(pipe: object) -> int:
    vae_scale = int(getattr(pipe, "vae_scale_factor_spatial", 8))
    transformer = getattr(pipe, "transformer", None)
    patch_size = getattr(getattr(transformer, "config", None), "patch_size", (1, 2, 2))
    if isinstance(patch_size, (list, tuple)) and len(patch_size) >= 3:
        return vae_scale * int(patch_size[1])
    return vae_scale * 2


def main() -> None:
    args = build_parser().parse_args()
    assert_wan_frame_count(args.num_frames)

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

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

    if args.lora_path:
        kwargs = {}
        if args.lora_weight_name:
            kwargs["weight_name"] = args.lora_weight_name
        if args.load_into_transformer_2:
            kwargs["load_into_transformer_2"] = True
        pipe.load_lora_weights(args.lora_path, adapter_name=args.adapter_name, **kwargs)
        pipe.set_adapters(args.adapter_name)

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    first_frame = load_rgb(args.image)
    multiple = maybe_pipeline_multiple(pipe)
    if args.height and args.width:
        height, width = args.height, args.width
        height = height // multiple * multiple
        width = width // multiple * multiple
    else:
        height, width = rounded_dimensions_for_image(first_frame, max_area=args.max_area, multiple=multiple)
    first_frame = resize_crop(first_frame, height=height, width=width)

    last_frame = None
    if args.last_image:
        last_frame = resize_crop(load_rgb(args.last_image), height=height, width=width)

    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    prompt = prompt_for_anime(
        args.prompt,
        aesthetic_score=args.aesthetic_score,
        motion_score=args.motion_score,
        append=not args.no_append_anime_clauses,
    )

    call_kwargs = {
        "image": first_frame,
        "prompt": prompt,
        "negative_prompt": args.negative_prompt or None,
        "height": height,
        "width": width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    if last_frame is not None:
        call_kwargs["last_image"] = last_frame
    if args.guidance_scale_2 is not None:
        call_kwargs["guidance_scale_2"] = args.guidance_scale_2

    output = pipe(**call_kwargs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(output.frames[0], str(output_path), fps=args.fps)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
