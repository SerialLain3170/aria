from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from .anita_production import TASKS, build_production_samples
from .manifest import read_jsonl
from .train_wan_i2v_lora import encode_prompt_like_wan, import_training_deps, validate_duration
from .train_wan_lora import (
    load_config,
    normalize_wan_latents,
    retrieve_latents,
    sample_flow_timesteps,
    save_lora,
    str_to_bool,
    weight_dtype,
)
from .video import assert_wan_frame_count, resize_and_crop, sample_indices

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TASK_PROMPTS = {
    "line_art": (
        "<TASK_LINE_ART> Generate a continuous anime production shot as clean line art only. "
        "Use black or dark lines on a plain light background. Do not add full color, lighting, "
        "painted backgrounds, texture, subtitles, or compositing effects."
    ),
    "character_color": (
        "<TASK_CHARACTER_COLOR> The condition is a line-art shot. Preserve the line art and motion, "
        "then add flat production colors to the main animated characters only. Keep background areas "
        "unpainted or unfinished. Do not return line art only."
    ),
    "compose_refine": (
        "<TASK_COMPOSE_REFINE> The condition is a character-colored shot. Preserve the characters, "
        "refine the drawing, color the background, and produce the final composited anime shot."
    ),
}


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def image_to_chw(
    path: str | Path,
    *,
    target_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if target_size is not None:
        height, width = target_size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    data = torch.from_numpy(np.array(image, dtype=np.uint8))
    return data.permute(2, 0, 1).contiguous()


def load_paths_tensor(
    paths: list[str],
    indices: list[int],
    *,
    target_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    if not paths:
        raise ValueError("paths must not be empty")
    return torch.stack([image_to_chw(paths[index], target_size=target_size) for index in indices])


def repeat_frame(frame: torch.Tensor, num_frames: int) -> torch.Tensor:
    return frame.unsqueeze(0).repeat(num_frames, 1, 1, 1)


def resize_sample_tensors(
    tensors: list[torch.Tensor],
    *,
    height: int,
    width: int,
    random_crop: bool,
) -> list[torch.Tensor]:
    stacked = torch.cat(tensors, dim=1)
    resized = resize_and_crop(stacked, height=height, width=width, random_crop=random_crop)
    out = []
    offset = 0
    for tensor in tensors:
        channels = tensor.shape[1]
        out.append(resized[:, offset : offset + channels])
        offset += channels
    return out


def normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float() / 127.5 - 1.0


def normalize_mask(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float() / 255.0


def parse_scene_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def record_matches_scenes(record: dict[str, Any], scenes: tuple[str, ...]) -> bool:
    # Match either a semantic sub-shot ("119_a_part000") or its whole parent scene ("119_a").
    return str(record.get("scene", "")) in scenes or str(record.get("parent_scene", "")) in scenes


def load_records(
    path_or_root: str | Path,
    *,
    tasks: tuple[str, ...],
    include_scenes: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    path = Path(path_or_root)
    records = read_jsonl(path) if path.is_file() else build_production_samples(path, tasks=tasks)
    enabled = set(tasks)
    records = [record for record in records if record.get("task") in enabled]
    if include_scenes:
        records = [record for record in records if record_matches_scenes(record, include_scenes)]
    return records


def prompt_for_record(record: dict[str, Any]) -> str:
    task = str(record.get("task", ""))
    parts = [TASK_PROMPTS.get(task, f"Task: {task}.")]
    prompt = str(record.get("prompt", "")).strip()
    caption = str(record.get("caption", "")).strip()
    if prompt:
        parts.append(prompt)
    if caption and caption not in prompt:
        parts.append(f"Shot content: {caption}")
    return " ".join(part for part in parts if part).strip()


def parse_task_weights(value: str | None, valid_tasks: tuple[str, ...] = TASKS) -> dict[str, float]:
    if not value:
        return {}
    weights: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid task weight {item!r}; expected task=weight")
        task, weight_text = item.split("=", 1)
        task = task.strip()
        if task not in valid_tasks:
            raise ValueError(f"unknown task in --task-weights: {task}")
        weight = float(weight_text)
        if weight <= 0:
            raise ValueError(f"task weight must be positive for {task}")
        weights[task] = weight
    return weights


def build_task_sample_weights(
    records: list[dict[str, Any]],
    *,
    tasks: tuple[str, ...],
    task_weights: dict[str, float] | None = None,
) -> list[float]:
    counts = {task: 0 for task in tasks}
    for record in records:
        task = str(record.get("task", ""))
        if task in counts:
            counts[task] += 1
    missing = [task for task, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"cannot task-balance dataset; no records for: {', '.join(missing)}")

    task_weights = task_weights or {}
    weights = []
    for record in records:
        task = str(record.get("task", ""))
        if task not in counts:
            weights.append(0.0)
            continue
        # Equalizes task probability by default. Optional user weights change the
        # task mixture without letting large tasks dominate by record count.
        weights.append(float(task_weights.get(task, 1.0)) / float(counts[task]))
    return weights


class WanAnitaProductionDataset(Dataset):
    def __init__(
        self,
        data: str | Path,
        *,
        num_frames: int,
        height: int,
        width: int,
        random_clip: bool,
        random_crop: bool,
        first_frame_prob: float,
        reference_prob: float,
        condition_dropout_prob: float,
        text_drop_prob: float,
        tasks: tuple[str, ...] = TASKS,
        include_scenes: tuple[str, ...] = (),
    ) -> None:
        self.records = load_records(data, tasks=tasks, include_scenes=include_scenes)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.random_clip = random_clip
        self.random_crop = random_crop
        self.first_frame_prob = first_frame_prob
        self.reference_prob = reference_prob
        self.condition_dropout_prob = condition_dropout_prob
        self.text_drop_prob = text_drop_prob

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target_paths = list(record.get("target_paths") or [])
        if not target_paths:
            raise ValueError(f"record has no target_paths: {record}")
        indices = sample_indices(len(target_paths), self.num_frames, random_clip=self.random_clip)
        source_size = image_size(target_paths[indices[0]])
        target = load_paths_tensor(target_paths, indices, target_size=source_size)
        source_height, source_width = source_size
        mask = torch.zeros(self.num_frames, 1, source_height, source_width, dtype=torch.uint8)
        source_kind = "none"

        primary_paths = list(record.get("primary_paths") or [])
        if primary_paths and random.random() >= self.condition_dropout_prob:
            condition = load_paths_tensor(primary_paths, indices, target_size=source_size)
            mask = torch.full_like(mask, 255)
            source_kind = "source_shot"
        elif random.random() < self.first_frame_prob:
            condition = torch.zeros_like(target)
            condition[0] = target[0]
            mask[0] = 255
            source_kind = "first_frame"
        elif random.random() < self.reference_prob and record.get("reference_paths"):
            reference_paths = list(record["reference_paths"])
            candidates = [idx for idx in range(len(reference_paths)) if idx not in set(indices)]
            if not candidates:
                candidates = list(range(len(reference_paths)))
            reference = image_to_chw(reference_paths[random.choice(candidates)], target_size=source_size)
            condition = repeat_frame(reference, self.num_frames)
            mask = torch.full_like(mask, 255)
            source_kind = "reference_image"
        else:
            condition = torch.zeros_like(target)

        condition, mask, target = resize_sample_tensors(
            [condition, mask, target],
            height=self.height,
            width=self.width,
            random_crop=self.random_crop,
        )
        prompt = prompt_for_record(record)
        if random.random() < self.text_drop_prob:
            prompt = TASK_PROMPTS.get(str(record.get("task", "")), "")
        return {
            "video": normalize_image(target).permute(1, 0, 2, 3).contiguous(),
            "condition_video": normalize_image(condition).permute(1, 0, 2, 3).contiguous(),
            "condition_mask": normalize_mask(mask).permute(1, 0, 2, 3).contiguous(),
            "caption": prompt,
            "task": record.get("task", ""),
            "source_kind": source_kind,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "video": torch.stack([item["video"] for item in batch]),
        "condition_video": torch.stack([item["condition_video"] for item in batch]),
        "condition_mask": torch.stack([item["condition_mask"] for item in batch]),
        "caption": [item["caption"] for item in batch],
        "task": [item["task"] for item in batch],
        "source_kind": [item["source_kind"] for item in batch],
    }


def prepare_video_condition(
    *,
    vae: torch.nn.Module,
    condition_video: torch.Tensor,
    condition_mask: torch.Tensor,
    dtype: torch.dtype,
    vae_sample_mode: str,
    vae_scale_factor_temporal: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    batch_size, _channels, num_frames, height, width = condition_video.shape
    latent_condition = retrieve_latents(
        vae.encode(condition_video.to(device=condition_video.device, dtype=vae.dtype)),
        sample_mode=vae_sample_mode,
    )
    latent_condition = normalize_wan_latents(vae, latent_condition).to(dtype)

    frame_mask = condition_mask.to(device=condition_video.device, dtype=dtype)
    frame_mask = frame_mask.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames,
        1,
        height,
        width,
    )
    frame_mask = F.interpolate(frame_mask, size=(latent_height, latent_width), mode="nearest")
    frame_mask = frame_mask.reshape(batch_size, num_frames, 1, latent_height, latent_width)
    frame_mask = frame_mask.permute(0, 2, 1, 3, 4).contiguous()

    first_frame_mask = torch.repeat_interleave(
        frame_mask[:, :, 0:1],
        dim=2,
        repeats=vae_scale_factor_temporal,
    )
    mask_lat_size = torch.concat([first_frame_mask, frame_mask[:, :, 1:]], dim=2)
    mask_lat_size = mask_lat_size.view(
        batch_size,
        -1,
        vae_scale_factor_temporal,
        latent_height,
        latent_width,
    )
    mask_lat_size = mask_lat_size.transpose(1, 2).to(latent_condition.device, dtype=dtype)
    return torch.concat([mask_lat_size, latent_condition], dim=1)


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(
        description="Train Wan2.2 I2V LoRA for Anita production stages."
    )
    parser.add_argument("--config")
    parser.add_argument("--data", required="data" not in defaults)
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="/data/shasegawa/t2a/models/Wan2.2-I2V-A14B-Diffusers",
    )
    parser.add_argument("--revision")
    parser.add_argument("--variant")
    parser.add_argument(
        "--output-dir",
        default="/data/shasegawa/t2a/outputs/wan22-anita-production-lora",
    )
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument(
        "--include-scenes",
        help="Optional comma-separated scene or parent_scene names to train on, for overfit tests.",
    )
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=5000)
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="to_q,to_k,to_v,to_out.0,ffn.net.0.proj,ffn.net.2",
    )
    parser.add_argument("--train-stage", choices=("high", "low"), default="high")
    parser.add_argument("--first-frame-prob", type=float, default=0.5)
    parser.add_argument("--reference-prob", type=float, default=0.5)
    parser.add_argument("--condition-dropout-prob", type=float, default=0.05)
    parser.add_argument("--text-drop-prob", type=float, default=0.05)
    parser.add_argument(
        "--task-sampling",
        choices=("natural", "balanced"),
        default="natural",
        help="balanced samples each task equally instead of by manifest frequency.",
    )
    parser.add_argument(
        "--task-weights",
        help="Optional comma-separated task mixture, for example line_art=1,character_color=1,compose_refine=2.",
    )
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gradient-checkpointing", type=str_to_bool, default=True)
    parser.add_argument("--enable-vae-tiling", type=str_to_bool, default=True)
    parser.add_argument("--vae-sample-mode", choices=("mean", "sample"), default="mean")
    parser.set_defaults(**defaults)
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)
    return build_parser(config).parse_args()


def main() -> None:
    args = parse_args()
    assert_wan_frame_count(args.num_frames)
    validate_duration(args.num_frames, args.fps)
    deps = import_training_deps()

    project_config = deps["ProjectConfiguration"](
        project_dir=args.output_dir,
        logging_dir=str(Path(args.output_dir) / "logs"),
    )
    accelerator = deps["Accelerator"](
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        project_config=project_config,
    )
    deps["set_seed"](args.seed)
    dtype = weight_dtype(args.mixed_precision)

    pipe = deps["WanImageToVideoPipeline"].from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=dtype,
    )
    if getattr(pipe.config, "expand_timesteps", False):
        raise ValueError("This trainer supports Wan I2V checkpoints without expand_timesteps only.")
    pipe.scheduler.set_timesteps(
        getattr(pipe.scheduler.config, "num_train_timesteps", 1000),
        device=accelerator.device,
    )

    vae = pipe.vae.to(accelerator.device, dtype=torch.float32)
    text_encoder = pipe.text_encoder.to(accelerator.device, dtype=dtype)
    tokenizer = pipe.tokenizer
    transformer = pipe.transformer
    transformer_2 = getattr(pipe, "transformer_2", None)
    image_encoder = getattr(pipe, "image_encoder", None)
    if args.train_stage == "low":
        if transformer_2 is None:
            raise ValueError("--train-stage low requires a Wan2.2 checkpoint with transformer_2")
        transformer = transformer_2

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if image_encoder is not None:
        image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    if transformer_2 is not None and transformer_2 is not transformer:
        transformer_2.requires_grad_(False)
    if args.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    if args.enable_vae_tiling and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
    transformer.add_adapter(
        deps["LoraConfig"](
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
    )
    transformer.train()

    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    train_dataset = WanAnitaProductionDataset(
        args.data,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        random_clip=True,
        random_crop=True,
        first_frame_prob=args.first_frame_prob,
        reference_prob=args.reference_prob,
        condition_dropout_prob=args.condition_dropout_prob,
        text_drop_prob=args.text_drop_prob,
        tasks=tasks,
        include_scenes=parse_scene_list(args.include_scenes),
    )
    if not train_dataset:
        raise ValueError(f"no Anita production records found in {args.data}")
    sampler = None
    shuffle = True
    if args.task_sampling == "balanced" or args.task_weights:
        sampler = WeightedRandomSampler(
            build_task_sample_weights(
                train_dataset.records,
                tasks=tasks,
                task_weights=parse_task_weights(args.task_weights),
            ),
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    trainable_params = [param for param in transformer.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_train_steps / max(steps_per_epoch, 1))
    lr_scheduler = deps["get_scheduler"](
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    transformer, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        transformer,
        optimizer,
        train_loader,
        lr_scheduler,
    )

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "training_args.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    metadata = {
        "base_model": args.pretrained_model_name_or_path,
        "dataset": "AnitaDataset",
        "task": "production-stage-video-to-video",
        "conditioning": "source_shot_or_optional_first_reference",
        "num_frames": str(args.num_frames),
        "fps": str(args.fps),
        "resolution": f"{args.width}x{args.height}",
        "train_stage": args.train_stage,
        "target_modules": args.target_modules,
    }

    latent_height = args.height // int(getattr(pipe, "vae_scale_factor_spatial", 8))
    latent_width = args.width // int(getattr(pipe, "vae_scale_factor_spatial", 8))
    vae_scale_factor_temporal = int(getattr(pipe, "vae_scale_factor_temporal", 4))
    boundary_ratio = getattr(pipe.config, "boundary_ratio", None)
    global_step = 0
    progress = tqdm(
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="wan22-anita-production",
    )

    for _epoch in range(num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(transformer):
                pixel_values = batch["video"].to(accelerator.device, dtype=torch.float32)
                condition_video = batch["condition_video"].to(
                    accelerator.device,
                    dtype=torch.float32,
                )
                condition_mask = batch["condition_mask"].to(accelerator.device, dtype=torch.float32)
                with torch.no_grad():
                    latents = retrieve_latents(
                        vae.encode(pixel_values),
                        sample_mode=args.vae_sample_mode,
                    )
                    latents = normalize_wan_latents(vae, latents).to(dtype)
                    condition = prepare_video_condition(
                        vae=vae,
                        condition_video=condition_video,
                        condition_mask=condition_mask,
                        dtype=dtype,
                        vae_sample_mode=args.vae_sample_mode,
                        vae_scale_factor_temporal=vae_scale_factor_temporal,
                        latent_height=latent_height,
                        latent_width=latent_width,
                    )
                    prompt_embeds = encode_prompt_like_wan(
                        tokenizer,
                        text_encoder,
                        batch["caption"],
                        accelerator.device,
                        dtype,
                        args.max_sequence_length,
                    )
                noise = torch.randn_like(latents)
                timesteps, sigmas = sample_flow_timesteps(
                    pipe.scheduler,
                    latents.shape[0],
                    accelerator.device,
                    train_stage=args.train_stage,
                    boundary_ratio=boundary_ratio,
                )
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                target = noise - latents
                latent_model_input = torch.cat([noisy_latents, condition], dim=1).to(dtype)
                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=None,
                    return_dict=False,
                )[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.detach().item():.4f}")
                if global_step % args.checkpointing_steps == 0:
                    save_lora(
                        accelerator=accelerator,
                        pipeline_cls=deps["WanImageToVideoPipeline"],
                        transformer=transformer,
                        output_dir=Path(args.output_dir) / f"checkpoint-{global_step}",
                        convert_state_dict_to_diffusers=deps["convert_state_dict_to_diffusers"],
                        get_peft_model_state_dict=deps["get_peft_model_state_dict"],
                        metadata=metadata,
                    )
                    accelerator.wait_for_everyone()
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    save_lora(
        accelerator=accelerator,
        pipeline_cls=deps["WanImageToVideoPipeline"],
        transformer=transformer,
        output_dir=args.output_dir,
        convert_state_dict_to_diffusers=deps["convert_state_dict_to_diffusers"],
        get_peft_model_state_dict=deps["get_peft_model_state_dict"],
        metadata=metadata,
    )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
