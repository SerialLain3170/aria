from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .anita_production import TASKS
from .infer_wan import DEFAULT_NEGATIVE_PROMPT
from .train_wan22_anita_production_lora import (
    image_size,
    load_paths_tensor,
    load_records,
    normalize_image,
    normalize_mask,
    parse_scene_list,
    prepare_video_condition,
    prompt_for_record,
    resize_sample_tensors,
)
from .train_wan_i2v_lora import encode_prompt_like_wan
from .video import assert_wan_frame_count, sample_indices

# Wan2.2 A14B two-expert sampling for Anita production stages, matching the conditioning used by
# train_wan22_anita_production_lora: the source shot (or first frame) is VAE-encoded and
# concatenated with a per-frame mask, and prompts come from the trainer's prompt_for_record.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render Anita production-stage shots with Wan2.2 I2V high/low LoRAs."
    )
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="/data/shasegawa/t2a/models/Wan2.2-I2V-A14B-Diffusers",
    )
    parser.add_argument("--data", required=True, help="Production manifest JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--high-lora", help="LoRA dir or .safetensors for the high-noise transformer.")
    parser.add_argument("--low-lora", help="LoRA dir or .safetensors for the low-noise transformer_2.")
    parser.add_argument("--high-lora-scale", type=float, default=1.0)
    parser.add_argument("--low-lora-scale", type=float, default=1.0)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--scenes", help="Comma-separated scene or parent_scene names to render.")
    parser.add_argument("--indices", help="Comma-separated manifest indices (after task filtering).")
    parser.add_argument(
        "--chain",
        action="store_true",
        help="Condition character_color/compose_refine on this run's previous generated stage "
        "instead of the ground-truth source shot.",
    )
    parser.add_argument(
        "--line-art-condition",
        choices=("first_frame", "none"),
        default="first_frame",
        help="Condition for line_art records, which have no source shot.",
    )
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--guidance-scale-2", type=float, help="Low-noise stage CFG; defaults to --guidance-scale.")
    parser.add_argument(
        "--flow-shift",
        type=float,
        help="Sampler flow shift. Defaults to the checkpoint scheduler config, which is what training uses.",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--high-device", default="cuda:0")
    parser.add_argument(
        "--low-device",
        help="Device for transformer_2. If equal to --high-device, the experts are swapped via CPU.",
    )
    return parser


def select_records(args: argparse.Namespace, tasks: tuple[str, ...]) -> list[tuple[int, dict[str, Any]]]:
    records = load_records(args.data, tasks=tasks, include_scenes=parse_scene_list(args.scenes))
    indexed = list(enumerate(records))
    if args.indices:
        wanted = {int(item) for item in args.indices.split(",") if item.strip()}
        indexed = [(index, record) for index, record in indexed if index in wanted]
    if not indexed:
        raise ValueError("no records selected; check --data, --tasks, --scenes, and --indices")
    return indexed


def load_teacher_forced_sample(
    record: dict[str, Any],
    *,
    num_frames: int,
    height: int,
    width: int,
    line_art_condition: str,
) -> dict[str, Any]:
    target_paths = list(record["target_paths"])
    indices = sample_indices(len(target_paths), num_frames, random_clip=False)
    source_size = image_size(target_paths[indices[0]])
    target = load_paths_tensor(target_paths, indices, target_size=source_size)
    mask = torch.zeros(num_frames, 1, *source_size, dtype=torch.uint8)
    primary_paths = list(record.get("primary_paths") or [])
    if primary_paths:
        condition = load_paths_tensor(primary_paths, indices, target_size=source_size)
        mask = torch.full_like(mask, 255)
        source_kind = "source_shot"
    elif line_art_condition == "first_frame":
        condition = torch.zeros_like(target)
        condition[0] = target[0]
        mask[0] = 255
        source_kind = "first_frame"
    else:
        condition = torch.zeros_like(target)
        source_kind = "none"
    # resize_and_crop returns float 0-255; keep uint8 so decoded outputs, metrics, and videos agree.
    condition, mask, target = (
        tensor.round().clamp(0, 255).to(torch.uint8)
        for tensor in resize_sample_tensors(
            [condition, mask, target],
            height=height,
            width=width,
            random_crop=False,
        )
    )
    return {
        "target": target,
        "condition": condition,
        "mask": mask,
        "source_kind": source_kind,
        "frame_indices": indices,
    }


def load_lora(
    pipe: Any,
    transformer: torch.nn.Module,
    path: str,
    *,
    adapter_name: str,
    scale: float,
) -> None:
    state_dict, metadata = pipe.lora_state_dict(path, return_lora_metadata=True)
    if metadata and "transformer.r" not in metadata:
        # Checkpoints saved before save_lora wrote the peft config (e.g. the r2 run) carry run
        # metadata in the LoRA config slot; ignore it and let diffusers infer rank from shapes.
        print(f"{path}: no LoRA config in checkpoint metadata; assuming lora_alpha == rank")
        metadata = None
    transformer.load_lora_adapter(
        state_dict,
        adapter_name=adapter_name,
        metadata=metadata,
        prefix="transformer",
    )
    if scale != 1.0:
        transformer.set_adapters([adapter_name], weights=[scale])


class ExpertPlacer:
    """Keeps each Wan2.2 expert on its device, swapping through CPU when both share one GPU."""

    def __init__(self, experts: dict[str, torch.nn.Module], devices: dict[str, torch.device]) -> None:
        self.experts = experts
        self.devices = devices
        self.shared = devices["high"] == devices["low"]
        self.active: str | None = None
        if not self.shared:
            for stage, model in experts.items():
                model.to(devices[stage])

    def get(self, stage: str) -> tuple[torch.nn.Module, torch.device]:
        if self.shared and self.active != stage:
            for other, model in self.experts.items():
                if other != stage:
                    model.to("cpu")
            torch.cuda.empty_cache()
            self.experts[stage].to(self.devices[stage])
            self.active = stage
        return self.experts[stage], self.devices[stage]


def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Direct GPU-to-GPU copies silently return zeros on this host even though CUDA reports peer
    # access, so tensors crossing between the high and low expert GPUs go through host memory.
    if tensor.device == device:
        return tensor
    if tensor.device.type == "cuda" and device.type == "cuda":
        return tensor.cpu().to(device)
    return tensor.to(device)


def to_model_video(tensor: torch.Tensor, device: torch.device, *, mask: bool = False) -> torch.Tensor:
    normalized = normalize_mask(tensor) if mask else normalize_image(tensor)
    return normalized.permute(1, 0, 2, 3).unsqueeze(0).to(device)


def decode_latents(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=torch.float32).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=torch.float32).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    # Inverse of normalize_wan_latents: (x - mean) / std.
    video = vae.decode((latents.float() * std + mean).to(vae.dtype), return_dict=False)[0]
    video = ((video[0].float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return video.permute(1, 0, 2, 3).cpu()


@torch.no_grad()
def sample_video(
    *,
    placer: ExpertPlacer,
    vae: torch.nn.Module,
    scheduler: Any,
    condition: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_embeds: torch.Tensor | None,
    args: argparse.Namespace,
    boundary_timestep: float | None,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    latent_frames = (args.num_frames - 1) // 4 + 1
    shape = (1, vae.config.z_dim, latent_frames, args.height // 8, args.width // 8)
    generator = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(condition.device)
    scheduler.set_timesteps(args.num_inference_steps, device=condition.device)
    guidance_2 = args.guidance_scale if args.guidance_scale_2 is None else args.guidance_scale_2
    for t in scheduler.timesteps:
        high = boundary_timestep is None or t >= boundary_timestep
        model, device = placer.get("high" if high else "low")
        guidance = args.guidance_scale if high else guidance_2
        model_input = to_device(torch.cat([latents, condition], dim=1).to(dtype), device)
        timestep = to_device(t.expand(1), device)
        noise_pred = model(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=to_device(prompt_embeds, device),
            encoder_hidden_states_image=None,
            return_dict=False,
        )[0]
        if negative_embeds is not None and guidance > 1.0:
            noise_uncond = model(
                hidden_states=model_input,
                timestep=timestep,
                encoder_hidden_states=to_device(negative_embeds, device),
                encoder_hidden_states_image=None,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance * (noise_pred - noise_uncond)
        noise_pred = to_device(noise_pred, latents.device).float()
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return decode_latents(vae, latents)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.float() / 255.0 - b.float() / 255.0) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * float(np.log10(1.0 / mse))


def saturation(video: torch.Tensor) -> float:
    rgb = video.float() / 255.0
    return (rgb.amax(dim=1) - rgb.amin(dim=1)).mean().item()


def motion(video: torch.Tensor) -> float:
    rgb = video.float() / 255.0
    return (rgb[1:] - rgb[:-1]).abs().mean().item() if rgb.shape[0] > 1 else 0.0


def video_metrics(generated: torch.Tensor, target: torch.Tensor, condition: torch.Tensor, source_kind: str) -> dict[str, float]:
    # If psnr_gen_vs_condition is high and psnr_gen_vs_target ~= psnr_condition_vs_target,
    # the model is copying its input instead of performing the task.
    metrics = {
        "psnr_gen_vs_target": psnr(generated, target),
        "saturation_gen": saturation(generated),
        "saturation_target": saturation(target),
        "motion_gen": motion(generated),
        "motion_target": motion(target),
    }
    if source_kind in {"source_shot", "chained"}:
        metrics.update(
            {
                "psnr_gen_vs_condition": psnr(generated, condition),
                "psnr_condition_vs_target": psnr(condition, target),
                "saturation_condition": saturation(condition),
            }
        )
    return metrics


def write_video(frames: torch.Tensor, path: Path, fps: int) -> None:
    # Pipe raw RGB frames to the system ffmpeg; the project venv has no OpenCV/imageio/PyAV.
    path.parent.mkdir(parents=True, exist_ok=True)
    _num_frames, _channels, height, width = frames.shape
    command = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path),
    ]
    if frames.dtype != torch.uint8:
        raise TypeError(f"write_video expects uint8 frames, got {frames.dtype}")
    raw = frames.permute(0, 2, 3, 1).contiguous().numpy().tobytes()
    subprocess.run(command, input=raw, check=True)


def hstack_videos(videos: list[torch.Tensor]) -> torch.Tensor:
    num_frames = min(video.shape[0] for video in videos)
    return torch.cat([video[:num_frames] for video in videos], dim=3)


def scene_key(record: dict[str, Any], index: int) -> str:
    return str(record.get("scene") or record.get("parent_scene") or f"record{index:04d}")


def main() -> None:
    args = build_parser().parse_args()
    assert_wan_frame_count(args.num_frames)
    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    selected = select_records(args, tasks)

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    dtype = torch.bfloat16
    high_device = torch.device(args.high_device)
    low_device = torch.device(args.low_device or args.high_device)
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        torch_dtype=dtype,
    )
    if pipe.transformer_2 is None or pipe.config.boundary_ratio is None:
        raise ValueError("this renderer expects a Wan2.2 A14B checkpoint with transformer_2 and boundary_ratio")
    if args.high_lora:
        load_lora(pipe, pipe.transformer, args.high_lora, adapter_name="high", scale=args.high_lora_scale)
    if args.low_lora:
        load_lora(pipe, pipe.transformer_2, args.low_lora, adapter_name="low", scale=args.low_lora_scale)
    pipe.transformer.eval()
    pipe.transformer_2.eval()

    scheduler_kwargs = {} if args.flow_shift is None else {"flow_shift": args.flow_shift}
    scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, **scheduler_kwargs)
    flow_shift = float(scheduler.config.flow_shift)
    boundary_timestep = pipe.config.boundary_ratio * scheduler.config.num_train_timesteps

    vae = pipe.vae.to(high_device)
    vae.enable_tiling()
    text_encoder = pipe.text_encoder.to(high_device, dtype=dtype)
    prompts = {index: prompt_for_record(record) for index, record in selected}
    with torch.no_grad():
        prompt_embeds = {
            index: encode_prompt_like_wan(
                pipe.tokenizer, text_encoder, [prompt], high_device, dtype, args.max_sequence_length
            )
            for index, prompt in prompts.items()
        }
        negative_embeds = None
        if args.negative_prompt:
            negative_embeds = encode_prompt_like_wan(
                pipe.tokenizer,
                text_encoder,
                [args.negative_prompt],
                high_device,
                dtype,
                args.max_sequence_length,
            )
    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    placer = ExpertPlacer(
        {"high": pipe.transformer, "low": pipe.transformer_2},
        {"high": high_device, "low": low_device},
    )

    # Render scene by scene in production order so --chain can feed each stage the previous output.
    by_scene: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in selected:
        by_scene[scene_key(record, index)].append((index, record))
    task_order = {task: order for order, task in enumerate(TASKS)}
    previous_task = {"character_color": "line_art", "compose_refine": "character_color"}

    output_dir = Path(args.output_dir)
    summary: list[dict[str, Any]] = []
    for scene, items in by_scene.items():
        items.sort(key=lambda item: task_order.get(str(item[1].get("task")), len(TASKS)))
        generated_by_task: dict[str, torch.Tensor] = {}
        for index, record in items:
            task = str(record["task"])
            sample = load_teacher_forced_sample(
                record,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                line_art_condition=args.line_art_condition,
            )
            condition, mask, source_kind = sample["condition"], sample["mask"], sample["source_kind"]
            upstream = previous_task.get(task)
            if args.chain and upstream in generated_by_task:
                condition = generated_by_task[upstream]
                mask = torch.full_like(mask, 255)
                source_kind = "chained"

            with torch.no_grad():
                latent_condition = prepare_video_condition(
                    vae=vae,
                    condition_video=to_model_video(condition, high_device),
                    condition_mask=to_model_video(mask, high_device, mask=True),
                    dtype=dtype,
                    vae_sample_mode="mean",
                    vae_scale_factor_temporal=int(pipe.vae_scale_factor_temporal),
                    latent_height=args.height // int(pipe.vae_scale_factor_spatial),
                    latent_width=args.width // int(pipe.vae_scale_factor_spatial),
                )
            seed = args.seed + index
            generated = sample_video(
                placer=placer,
                vae=vae,
                scheduler=scheduler,
                condition=latent_condition,
                prompt_embeds=prompt_embeds[index],
                negative_embeds=negative_embeds,
                args=args,
                boundary_timestep=boundary_timestep,
                dtype=dtype,
                seed=seed,
            )
            generated_by_task[task] = generated

            stem = output_dir / scene / f"{task}_idx{index:04d}"
            generated_path = stem.with_name(stem.name + "_generated.mp4")
            comparison_path = stem.with_name(stem.name + "_condition_generated_target.mp4")
            write_video(generated, generated_path, args.fps)
            write_video(hstack_videos([condition, generated, sample["target"]]), comparison_path, args.fps)
            info = {
                "scene": scene,
                "task": task,
                "manifest_index": index,
                "source_kind": source_kind,
                "prompt": prompts[index],
                "seed": seed,
                "frame_indices": sample["frame_indices"],
                "generated_path": str(generated_path),
                "comparison_path": str(comparison_path),
                "metrics": video_metrics(generated, sample["target"], condition, source_kind),
            }
            stem.with_name(stem.name + "_render.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            summary.append(info)
            print(f"{scene} {task}: {json.dumps(info['metrics'])}")

        if all(task in generated_by_task for task in TASKS):
            progression_path = output_dir / scene / "line_art_character_color_compose_refine_generated.mp4"
            write_video(hstack_videos([generated_by_task[task] for task in TASKS]), progression_path, args.fps)

    settings = dict(vars(args), flow_shift_used=flow_shift, boundary_timestep=float(boundary_timestep))
    (output_dir / "render_summary.json").write_text(
        json.dumps({"settings": settings, "renders": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'render_summary.json'}")


if __name__ == "__main__":
    main()
