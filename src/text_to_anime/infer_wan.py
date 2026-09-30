from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .video import assert_wan_frame_count


DEFAULT_NEGATIVE_PROMPT = (
    "subtitles, captions, text, watermark, logo, credits, low quality, jpeg artifacts, blurry, "
    "distorted hands, malformed face, duplicate character, flicker"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Wan/AniSora text-to-anime inference.")
    parser.add_argument("--pretrained-model-name-or-path", default="Wan-AI/Wan2.2-T2V-14B-Diffusers")
    parser.add_argument("--revision")
    parser.add_argument("--variant")
    parser.add_argument("--lora-path")
    parser.add_argument("--lora-weight-name")
    parser.add_argument("--adapter-name", default="anime")
    parser.add_argument("--load-into-transformer-2", action="store_true")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale-2", type=float)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--aesthetic-score", type=float, default=5.5)
    parser.add_argument("--motion-score", type=float, default=2.5)
    parser.add_argument("--no-append-anime-clauses", action="store_true")
    return parser


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def prompt_for_anime(prompt: str, *, aesthetic_score: float, motion_score: float, append: bool) -> str:
    if not append:
        return prompt
    clauses = [
        "hand-drawn Japanese cel animation",
        f"aesthetic score: {aesthetic_score:.1f}.",
        f"motion score: {motion_score:.1f}.",
        "There is no text in the video.",
    ]
    lower_prompt = prompt.lower()
    additions = [clause for clause in clauses if clause.lower().rstrip(".") not in lower_prompt]
    return f"{prompt.strip()} {' '.join(additions)}".strip()


def main() -> None:
    args = build_parser().parse_args()
    assert_wan_frame_count(args.num_frames)

    from diffusers import AutoencoderKLWan, WanPipeline
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
    pipe = WanPipeline.from_pretrained(
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

    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    prompt = prompt_for_anime(
        args.prompt,
        aesthetic_score=args.aesthetic_score,
        motion_score=args.motion_score,
        append=not args.no_append_anime_clauses,
    )

    output = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt or None,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        generator=generator,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(output.frames[0], str(output_path), fps=args.fps)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

