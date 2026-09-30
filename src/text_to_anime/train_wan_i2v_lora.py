from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .captioning import build_caption
from .manifest import read_jsonl
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


def image_to_chw(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    data = torch.from_numpy(np.array(image, dtype=np.uint8))
    return data.permute(2, 0, 1).contiguous()


def load_sequence_tensor(
    frame_dir: str | Path,
    *,
    num_frames: int,
    height: int,
    width: int,
    random_clip: bool,
    random_crop: bool,
) -> torch.Tensor:
    frame_paths = sorted(path for path in Path(frame_dir).iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not frame_paths:
        raise ValueError(f"no image frames found in {frame_dir}")
    indices = sample_indices(len(frame_paths), num_frames, random_clip=random_clip)
    frames = torch.stack([image_to_chw(frame_paths[index]) for index in indices])
    frames = resize_and_crop(frames, height=height, width=width, random_crop=random_crop)
    frames = frames / 127.5 - 1.0
    return frames.permute(1, 0, 2, 3).contiguous()


class AnitaI2VDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        num_frames: int,
        height: int,
        width: int,
        random_clip: bool,
        random_crop: bool,
    ) -> None:
        self.records = [record for record in read_jsonl(manifest_path) if record.get("frame_dir")]
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.random_clip = random_clip
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        video = load_sequence_tensor(
            record["frame_dir"],
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            random_clip=self.random_clip,
            random_crop=self.random_crop,
        )
        return {
            "video": video,
            "first_frame": video[:, 0].contiguous(),
            "caption": build_caption(record),
            "frame_dir": record["frame_dir"],
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "video": torch.stack([item["video"] for item in batch]),
        "first_frame": torch.stack([item["first_frame"] for item in batch]),
        "caption": [item["caption"] for item in batch],
        "frame_dir": [item["frame_dir"] for item in batch],
    }


def validate_duration(num_frames: int, fps: int) -> None:
    duration = num_frames / fps
    if duration < 3.0 or duration > 5.0:
        raise ValueError(f"expected 3-5 second clips, got {duration:.2f}s from {num_frames} frames at {fps} fps")


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Train AnitaDataset image-to-video LoRA for Wan I2V.")
    parser.set_defaults(**defaults)

    parser.add_argument("--config")
    parser.add_argument("--manifest", required="manifest" not in defaults)
    parser.add_argument("--validation-manifest")
    parser.add_argument("--pretrained-model-name-or-path", default="/data/shasegawa/t2a/models/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--revision")
    parser.add_argument("--variant")
    parser.add_argument("--output-dir", default="/data/shasegawa/t2a/outputs/wan22-i2v-anita-lora")

    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-sequence-length", type=int, default=512)

    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=3000)
    parser.add_argument("--checkpointing-steps", type=int, default=250)
    parser.add_argument("--validation-steps", type=int, default=500)
    parser.add_argument("--num-validation-samples", type=int, default=50)

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
        help="Attention projections plus FFN projections, which carry temporal dynamics in Wan's 3D blocks.",
    )
    parser.add_argument("--train-stage", choices=("high", "low"), default="high")

    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gradient-checkpointing", type=str_to_bool, default=True)
    parser.add_argument("--enable-vae-tiling", type=str_to_bool, default=True)
    parser.add_argument("--vae-sample-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--negative-prompt", default="subtitles, captions, text, watermark, logo, credits")
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)
    parser = build_parser(config)
    args = parser.parse_args()
    if not 50 <= int(args.num_validation_samples) <= 100:
        raise ValueError("--num-validation-samples must be between 50 and 100 fixed first-frame/prompt pairs")
    return args


def import_training_deps():
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.optimization import get_scheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    return {
        "Accelerator": Accelerator,
        "ProjectConfiguration": ProjectConfiguration,
        "set_seed": set_seed,
        "get_logger": get_logger,
        "AutoencoderKLWan": AutoencoderKLWan,
        "WanImageToVideoPipeline": WanImageToVideoPipeline,
        "get_scheduler": get_scheduler,
        "convert_state_dict_to_diffusers": convert_state_dict_to_diffusers,
        "LoraConfig": LoraConfig,
        "get_peft_model_state_dict": get_peft_model_state_dict,
    }


def encode_prompt_like_wan(tokenizer: Any, text_encoder: Any, captions: list[str], device: torch.device, dtype: torch.dtype, max_length: int) -> torch.Tensor:
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids, mask).last_hidden_state.to(dtype=dtype, device=device)
    prompt_embeds = [embeds[:seq_len] for embeds, seq_len in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([embeds, embeds.new_zeros(max_length - embeds.size(0), embeds.size(1))]) for embeds in prompt_embeds],
        dim=0,
    )
    return prompt_embeds


def prepare_first_frame_condition(
    *,
    vae: torch.nn.Module,
    first_frame: torch.Tensor,
    num_frames: int,
    dtype: torch.dtype,
    vae_sample_mode: str,
    vae_scale_factor_temporal: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    batch_size, channels, height, width = first_frame.shape
    image = first_frame.unsqueeze(2)
    video_condition = torch.cat(
        [image, image.new_zeros(batch_size, channels, num_frames - 1, height, width)], dim=2
    ).to(device=first_frame.device, dtype=vae.dtype)
    latent_condition = retrieve_latents(vae.encode(video_condition), sample_mode=vae_sample_mode)
    latent_condition = normalize_wan_latents(vae, latent_condition).to(dtype)

    mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width, device=first_frame.device, dtype=dtype)
    mask_lat_size[:, :, list(range(1, num_frames))] = 0
    first_frame_mask = torch.repeat_interleave(mask_lat_size[:, :, 0:1], dim=2, repeats=vae_scale_factor_temporal)
    mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
    mask_lat_size = mask_lat_size.view(batch_size, -1, vae_scale_factor_temporal, latent_height, latent_width)
    mask_lat_size = mask_lat_size.transpose(1, 2).to(latent_condition.device, dtype=dtype)
    return torch.concat([mask_lat_size, latent_condition], dim=1)


def first_frame_pil_from_tensor(frame: torch.Tensor) -> Image.Image:
    pixels = ((frame.detach().cpu().float().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


def run_validation(
    *,
    args: argparse.Namespace,
    deps: dict[str, Any],
    accelerator: Any,
    transformer: torch.nn.Module,
    pipeline_components: dict[str, Any],
    step: int,
    logger: Any,
) -> None:
    if not accelerator.is_main_process or not args.validation_manifest:
        return
    records = read_jsonl(args.validation_manifest)[: args.num_validation_samples]
    if not records:
        return

    from diffusers.utils import export_to_video

    validation_dir = Path(args.output_dir) / "validation" / f"step_{step:06d}"
    validation_dir.mkdir(parents=True, exist_ok=True)
    pipe = deps["WanImageToVideoPipeline"].from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=pipeline_components["tokenizer"],
        text_encoder=pipeline_components["text_encoder"],
        transformer=accelerator.unwrap_model(transformer),
        transformer_2=pipeline_components.get("transformer_2"),
        vae=pipeline_components["vae"],
        scheduler=pipeline_components["scheduler"],
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype(args.mixed_precision),
    )
    pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + step)

    for index, record in enumerate(records):
        frame_dir = record.get("frame_dir")
        if not frame_dir:
            continue
        sample = load_sequence_tensor(
            frame_dir,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            random_clip=False,
            random_crop=False,
        )
        image = first_frame_pil_from_tensor(sample[:, 0])
        with torch.no_grad():
            frames = pipe(
                image=image,
                prompt=build_caption(record),
                negative_prompt=args.negative_prompt or None,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=8,
                guidance_scale=5.0,
                generator=generator,
            ).frames[0]
        export_to_video(frames, str(validation_dir / f"{index:03d}.mp4"), fps=args.fps)
    logger.info("wrote %s Anita I2V validation samples to %s", len(records), validation_dir)


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
    logger = deps["get_logger"](__name__, log_level="INFO")
    deps["set_seed"](args.seed)

    dtype = weight_dtype(args.mixed_precision)
    pipe = deps["WanImageToVideoPipeline"].from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=dtype,
    )
    if getattr(pipe.config, "expand_timesteps", False):
        raise ValueError("This trainer currently supports Wan I2V checkpoints without expand_timesteps only.")
    pipe.scheduler.set_timesteps(getattr(pipe.scheduler.config, "num_train_timesteps", 1000), device=accelerator.device)

    vae = pipe.vae.to(accelerator.device, dtype=torch.float32)
    text_encoder = pipe.text_encoder.to(accelerator.device, dtype=dtype)
    tokenizer = pipe.tokenizer
    transformer = pipe.transformer
    transformer_2 = getattr(pipe, "transformer_2", None)
    image_encoder = getattr(pipe, "image_encoder", None)

    if args.train_stage == "low":
        if transformer_2 is None:
            raise ValueError("--train-stage low requires a Wan 2.2 I2V checkpoint with transformer_2")
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
    lora_config = deps["LoraConfig"](
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(lora_config)
    transformer.train()

    train_dataset = AnitaI2VDataset(
        args.manifest,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        random_clip=True,
        random_crop=True,
    )
    if not train_dataset:
        raise ValueError(f"no Anita frame_dir records found in {args.manifest}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
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
        transformer, optimizer, train_loader, lr_scheduler
    )

    metadata = {
        "base_model": args.pretrained_model_name_or_path,
        "dataset": "AnitaDataset",
        "task": "image-to-video",
        "conditioning": "first_frame",
        "num_frames": str(args.num_frames),
        "fps": str(args.fps),
        "resolution": f"{args.width}x{args.height}",
        "train_stage": args.train_stage,
        "target_modules": args.target_modules,
    }
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "training_args.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
        )

    latent_height = args.height // int(getattr(pipe, "vae_scale_factor_spatial", 8))
    latent_width = args.width // int(getattr(pipe, "vae_scale_factor_spatial", 8))
    vae_scale_factor_temporal = int(getattr(pipe, "vae_scale_factor_temporal", 4))
    boundary_ratio = getattr(pipe.config, "boundary_ratio", None)
    global_step = 0
    progress = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process, desc="anita-i2v")

    for _epoch in range(num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(transformer):
                pixel_values = batch["video"].to(accelerator.device, dtype=torch.float32)
                first_frame = batch["first_frame"].to(accelerator.device, dtype=torch.float32)
                with torch.no_grad():
                    latents = retrieve_latents(vae.encode(pixel_values), sample_mode=args.vae_sample_mode)
                    latents = normalize_wan_latents(vae, latents).to(dtype)
                    condition = prepare_first_frame_condition(
                        vae=vae,
                        first_frame=first_frame,
                        num_frames=args.num_frames,
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
                progress.set_postfix(loss=f"{loss.detach().item():.4f}", lr=lr_scheduler.get_last_lr()[0])

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

                if args.validation_steps and global_step % args.validation_steps == 0:
                    run_validation(
                        args=args,
                        deps=deps,
                        accelerator=accelerator,
                        transformer=transformer,
                        pipeline_components={
                            "tokenizer": tokenizer,
                            "text_encoder": text_encoder,
                            "transformer_2": transformer_2 if args.train_stage == "high" else None,
                            "vae": vae,
                            "scheduler": pipe.scheduler,
                        },
                        step=global_step,
                        logger=logger,
                    )

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
