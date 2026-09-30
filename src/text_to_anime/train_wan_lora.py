from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .captioning import build_caption
from .manifest import read_jsonl
from .video import assert_wan_frame_count, load_video_tensor


class AnimeVideoDataset(Dataset):
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
        self.records = read_jsonl(manifest_path)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.random_clip = random_clip
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        video = load_video_tensor(
            record["video_path"],
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            random_clip=self.random_clip,
            random_crop=self.random_crop,
        )
        return {"video": video, "caption": build_caption(record), "path": record["video_path"]}


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "video": torch.stack([item["video"] for item in batch]),
        "caption": [item["caption"] for item in batch],
        "path": [item["path"] for item in batch],
    }


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(handle) or {}
        else:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config must contain an object: {path}")
    return data


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "on"}


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Train a Wan/AniSora text-to-video LoRA.")
    parser.set_defaults(**defaults)

    parser.add_argument("--config")
    parser.add_argument("--manifest", required="manifest" not in defaults)
    parser.add_argument("--validation-manifest")
    parser.add_argument("--pretrained-model-name-or-path", default="Wan-AI/Wan2.2-T2V-14B-Diffusers")
    parser.add_argument("--revision")
    parser.add_argument("--variant")
    parser.add_argument("--output-dir", default="/data/shasegawa/t2a/outputs/anisora-v32-animeshooter-vn-lora")

    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-sequence-length", type=int, default=512)

    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=10000)
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--validation-steps", type=int, default=500)
    parser.add_argument("--validation-prompts", default="configs/validation_prompts.txt")

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=250)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--train-stage", choices=("high", "low"), default="high")

    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gradient-checkpointing", type=str_to_bool, default=True)
    parser.add_argument("--enable-vae-tiling", type=str_to_bool, default=True)
    parser.add_argument("--vae-sample-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--negative-prompt", default="")
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)
    parser = build_parser(config)
    return parser.parse_args()


def import_training_deps():
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import AutoencoderKLWan, WanPipeline
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
        "WanPipeline": WanPipeline,
        "get_scheduler": get_scheduler,
        "convert_state_dict_to_diffusers": convert_state_dict_to_diffusers,
        "LoraConfig": LoraConfig,
        "get_peft_model_state_dict": get_peft_model_state_dict,
    }


def weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def retrieve_latents(encoded: Any, *, sample_mode: str) -> torch.Tensor:
    latent_dist = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
    if sample_mode == "sample":
        return latent_dist.sample()
    if hasattr(latent_dist, "mode"):
        return latent_dist.mode()
    return latent_dist.mean


def normalize_wan_latents(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=torch.float32).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    std = 1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=torch.float32).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    return (latents.float() - mean) * std


def encode_prompt(tokenizer: Any, text_encoder: Any, captions: list[str], device: torch.device, max_length: int) -> torch.Tensor:
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    prompt_embeds = text_encoder(**text_inputs).last_hidden_state
    return prompt_embeds


def sample_flow_timesteps(
    scheduler: Any,
    batch_size: int,
    device: torch.device,
    *,
    train_stage: str,
    boundary_ratio: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps = scheduler.timesteps.to(device)
    sigmas = scheduler.sigmas.to(device)
    usable = sigmas.shape[0] - 1 if sigmas.shape[0] == timesteps.shape[0] + 1 else sigmas.shape[0]
    timesteps = timesteps[:usable]
    sigmas = sigmas[:usable]

    if boundary_ratio is not None:
        boundary = boundary_ratio * float(getattr(scheduler.config, "num_train_timesteps", 1000))
        if train_stage == "high":
            candidates = torch.nonzero(timesteps >= boundary, as_tuple=False).flatten()
        else:
            candidates = torch.nonzero(timesteps < boundary, as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError(f"no scheduler timesteps available for train_stage={train_stage}")
    else:
        candidates = torch.arange(timesteps.shape[0], device=device)

    chosen = candidates[torch.randint(0, candidates.numel(), (batch_size,), device=device)]
    sigma = sigmas[chosen].flatten()
    while sigma.ndim < 5:
        sigma = sigma.unsqueeze(-1)
    return timesteps[chosen], sigma


def save_lora(
    *,
    accelerator: Any,
    pipeline_cls: Any,
    transformer: torch.nn.Module,
    output_dir: str | Path,
    convert_state_dict_to_diffusers: Any,
    get_peft_model_state_dict: Any,
    metadata: dict[str, str],
) -> None:
    if not accelerator.is_main_process:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(transformer)
    state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped))
    # Diffusers reads transformer_lora_adapter_metadata back as the LoraConfig, so it must hold the
    # peft config (rank, alpha, targets). Descriptive run metadata goes in a sidecar JSON instead.
    pipeline_cls.save_lora_weights(
        save_directory=output_dir,
        transformer_lora_layers=state_dict,
        safe_serialization=True,
        transformer_lora_adapter_metadata=unwrapped.peft_config["default"].to_dict(),
    )
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


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
    if not accelerator.is_main_process:
        return
    prompt_path = Path(args.validation_prompts) if args.validation_prompts else Path()

    prompts = []
    if prompt_path.exists():
        prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts and args.validation_manifest:
        prompts = [build_caption(record) for record in read_jsonl(args.validation_manifest)[:4]]
    if not prompts:
        return

    validation_dir = Path(args.output_dir) / "validation" / f"step_{step:06d}"
    validation_dir.mkdir(parents=True, exist_ok=True)

    from diffusers.utils import export_to_video

    pipeline = deps["WanPipeline"].from_pretrained(
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
    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + step)
    for index, prompt in enumerate(prompts[:4]):
        with torch.no_grad():
            frames = pipeline(
                prompt=prompt,
                negative_prompt=args.negative_prompt or None,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=8,
                guidance_scale=1.0,
                generator=generator,
            ).frames[0]
        export_to_video(frames, str(validation_dir / f"{index:02d}.mp4"), fps=args.fps)
    logger.info("wrote validation samples to %s", validation_dir)


def main() -> None:
    args = parse_args()
    assert_wan_frame_count(args.num_frames)
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
    pipe = deps["WanPipeline"].from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=dtype,
    )
    pipe.scheduler.set_timesteps(getattr(pipe.scheduler.config, "num_train_timesteps", 1000), device=accelerator.device)

    vae = pipe.vae.to(accelerator.device, dtype=torch.float32)
    text_encoder = pipe.text_encoder.to(accelerator.device, dtype=dtype)
    tokenizer = pipe.tokenizer
    transformer = pipe.transformer
    transformer_2 = getattr(pipe, "transformer_2", None)

    if args.train_stage == "low":
        if transformer_2 is None:
            raise ValueError("--train-stage low requires a Wan 2.2 style checkpoint with transformer_2")
        transformer = transformer_2

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
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

    train_dataset = AnimeVideoDataset(
        args.manifest,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        random_clip=True,
        random_crop=True,
    )
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
        "num_frames": str(args.num_frames),
        "resolution": f"{args.width}x{args.height}",
        "train_stage": args.train_stage,
    }
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "training_args.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    global_step = 0
    progress = tqdm(
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="training",
    )

    boundary_ratio = getattr(pipe.config, "boundary_ratio", None)
    for _epoch in range(num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(transformer):
                pixel_values = batch["video"].to(accelerator.device, dtype=torch.float32)
                with torch.no_grad():
                    latents = retrieve_latents(vae.encode(pixel_values), sample_mode=args.vae_sample_mode)
                    latents = normalize_wan_latents(vae, latents).to(dtype)
                    prompt_embeds = encode_prompt(
                        tokenizer,
                        text_encoder,
                        batch["caption"],
                        accelerator.device,
                        args.max_sequence_length,
                    ).to(dtype)

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

                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
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
                        pipeline_cls=deps["WanPipeline"],
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
        pipeline_cls=deps["WanPipeline"],
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

