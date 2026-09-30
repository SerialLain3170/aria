from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from .anita_production import iter_shot_sets
from .manifest import read_jsonl
from .train_wan22_anita_production_lora import build_task_sample_weights, parse_task_weights
from .train_wan_i2v_lora import encode_prompt_like_wan
from .train_wan_lora import (
    load_config,
    normalize_wan_latents,
    retrieve_latents,
    sample_flow_timesteps,
    save_lora,
    str_to_bool,
    weight_dtype,
)

# Wan2.1 VACE LoRA training on AnitaDataset production stages. Each task is posed the way VACE was
# pretrained (control video + mask where 1 = generate, plus optional reference images), so the
# LoRA adapts an existing skill instead of learning a new conditioning scheme:
#   inbetween        line-art keyframes (mask 0) -> full line-art sequence
#   character_color  line-art video (mask 1) + one colored reference drawing -> character colors
#   compose_refine   character layer over the recovered background (mask 1) [+ plate ref] -> final shot
# Conditioning latents are built with WanVACEPipeline's own prepare_video_latents/prepare_masks so
# stock WanVACEPipeline inference sees exactly the training conditioning.

TASKS = ("inbetween", "character_color", "compose_refine")
TIMELINE_FPS = 24
TASK_PROMPTS = {
    "inbetween": (
        "Clean anime production line art, dark pencil lines on plain white paper. "
        "In-between animation drawings that move smoothly between the given key drawings."
    ),
    "character_color": (
        "Flat anime production colors on the characters, matching the color reference exactly. "
        "Keep every line of the line art, plain white background."
    ),
    "compose_refine": (
        "Final composited anime shot: the colored characters integrated into the painted background, "
        "with finished lighting and effects."
    ),
}


def timeline_from_ids(drawing_ids: list[str]) -> list[str]:
    """Expand timing-sheet frame numbers into one drawing per timeline frame, holding each drawing."""
    try:
        numbered = sorted((int(frame_id), frame_id) for frame_id in drawing_ids)
    except ValueError:
        return sorted(drawing_ids)
    timeline = []
    for (number, frame_id), (next_number, _next_id) in zip(numbered, numbered[1:] + [(numbered[-1][0] + 1, "")]):
        timeline.extend([frame_id] * (next_number - number))
    return timeline


def wan_frame_count(available: int, max_frames: int) -> int:
    count = min(available, max_frames)
    return (count - 1) // 4 * 4 + 1 if count >= 1 else 0


def load_captions(path: str | None) -> dict[str, list[tuple[set[str], str]]]:
    captions: dict[str, list[tuple[set[str], str]]] = {}
    if not path or not Path(path).is_file():
        return captions
    seen = set()
    for record in read_jsonl(path):
        caption = str(record.get("caption", "")).strip()
        key = f"{record.get('work')}/{record.get('parent_scene') or record.get('scene')}"
        frame_ids = tuple(record.get("frame_ids") or [])
        if not caption or (key, frame_ids) in seen:
            continue
        seen.add((key, frame_ids))
        captions.setdefault(key, []).append((set(frame_ids), caption))
    return captions


def caption_for(captions: dict[str, list[tuple[set[str], str]]], key: str, drawing_ids: list[str]) -> str:
    candidates = captions.get(key) or []
    if not candidates:
        return ""
    wanted = set(drawing_ids)
    return max(candidates, key=lambda item: len(item[0] & wanted))[1]


def build_clip_sources(
    root: str | Path,
    *,
    tasks: tuple[str, ...],
    plates_index: str | None,
    min_frames: int,
    timeline_stride: int,
) -> list[dict[str, Any]]:
    plates = {}
    if plates_index and Path(plates_index).is_file():
        plates = {f"{row['work']}/{row['scene']}": row for row in read_jsonl(plates_index)}
    sources = []
    for shot in iter_shot_sets(root):
        key = f"{shot.work}/{shot.scene}"
        candidates = {
            "inbetween": sorted(shot.sketch_paths),
            "character_color": sorted(set(shot.sketch_paths) & set(shot.color_paths)),
            "compose_refine": sorted(set(shot.color_paths) & set(shot.composition_paths)),
        }
        for task in tasks:
            drawing_ids = candidates[task]
            plate = plates.get(key)
            if task == "compose_refine":
                if plate is None:
                    continue
                drawing_ids = [frame_id for frame_id in drawing_ids if frame_id in plate["background_paths"]]
            if not drawing_ids:
                continue
            timeline = timeline_from_ids(drawing_ids)
            if (len(timeline) - 1) // timeline_stride + 1 < min_frames:
                continue
            sources.append(
                {
                    "key": key,
                    "work": shot.work,
                    "scene": shot.scene,
                    "task": task,
                    "timeline": timeline,
                    "drawing_ids": drawing_ids,
                    "sketch_paths": shot.sketch_paths,
                    "color_paths": shot.color_paths,
                    "composition_paths": shot.composition_paths,
                    "background_paths": plate["background_paths"] if plate else {},
                    "plate_path": plate["plate_path"] if plate else None,
                    "plate_static": bool(plate["static"]) if plate else False,
                }
            )
    return sources


def split_sources(
    sources: list[dict[str, Any]],
    *,
    val_scenes: tuple[str, ...],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Split by whole shot so no task of a validation shot is ever trained on.
    def is_val(source: dict[str, Any]) -> bool:
        if source["key"] in val_scenes or source["scene"] in val_scenes:
            return True
        digest = hashlib.sha1(f"{seed}:{source['key']}".encode()).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF < val_ratio

    train = [source for source in sources if not is_val(source)]
    val = [source for source in sources if is_val(source)]
    return train, val


def open_over_white(path: str | Path) -> Image.Image:
    image = Image.open(path)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, image).convert("RGB")
    return image.convert("RGB")


def cover_crop(image: Image.Image, *, height: int, width: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(width, round(image.width * scale)), max(height, round(image.height * scale))),
        Image.Resampling.LANCZOS,
        reducing_gap=3.0,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def reference_image(path: str | Path, *, height: int, width: int) -> Image.Image:
    """Reference drawing on white, fitted inside the target box (WanVACEPipeline then letterboxes it)."""
    image = open_over_white(path)
    scale = min(width / image.width, height / image.height, 1.0)
    size = (max(8, int(image.width * scale) // 8 * 8), max(8, int(image.height * scale) // 8 * 8))
    return image.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def letterbox_like_vace(image: Image.Image, *, height: int, width: int) -> torch.Tensor:
    """Replicates WanVACEPipeline.preprocess_conditions for one reference image, in [-1, 1]."""
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
    img_height, img_width = tensor.shape[-2:]
    scale = min(height / img_height, width / img_width)
    new_height, new_width = int(img_height * scale), int(img_width * scale)
    resized = F.interpolate(tensor[None], size=(new_height, new_width), mode="bilinear", align_corners=False)[0]
    canvas = torch.ones(3, height, width)
    top, left = (height - new_height) // 2, (width - new_width) // 2
    canvas[:, top : top + new_height, left : left + new_width] = resized
    return canvas


class FrameCache:
    def __init__(self, *, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.cache: dict[tuple[str, str], np.ndarray] = {}

    def rgb(self, path: str) -> np.ndarray:
        key = ("rgb", path)
        if key not in self.cache:
            self.cache[key] = np.asarray(cover_crop(open_over_white(path), height=self.height, width=self.width))
        return self.cache[key]

    def rgba(self, path: str) -> np.ndarray:
        key = ("rgba", path)
        if key not in self.cache:
            image = Image.open(path).convert("RGBA")
            self.cache[key] = np.asarray(cover_crop(image, height=self.height, width=self.width))
        return self.cache[key]


def window_drawings(
    timeline: list[str],
    *,
    max_frames: int,
    stride: int,
    rng: random.Random,
    center: bool,
) -> list[str]:
    available = (len(timeline) - 1) // stride + 1
    num_frames = wan_frame_count(available, max_frames)
    span = (num_frames - 1) * stride + 1
    slack = len(timeline) - span
    start = slack // 2 if center else rng.randint(0, slack)
    return [timeline[start + index * stride] for index in range(num_frames)]


def keyframe_indices(num_frames: int, keyframe_stride: int) -> list[int]:
    keys = list(range(0, num_frames, keyframe_stride))
    if keys[-1] != num_frames - 1:
        keys.append(num_frames - 1)
    return keys


def build_sample(
    source: dict[str, Any],
    *,
    height: int,
    width: int,
    max_frames: int,
    timeline_stride: int,
    keyframe_stride: int,
    compose_plate_reference: bool,
    rng: random.Random,
    center: bool,
) -> dict[str, Any]:
    """Returns uint8 frames (T, 3, H, W), a {0,1} generate-mask (T, 1, H, W), and an optional reference PIL."""
    cache = FrameCache(height=height, width=width)
    drawings = window_drawings(
        source["timeline"], max_frames=max_frames, stride=timeline_stride, rng=rng, center=center
    )
    task = source["task"]
    reference = None
    if task == "inbetween":
        target = np.stack([cache.rgb(source["sketch_paths"][frame_id]) for frame_id in drawings])
        control = np.full_like(target, 128)
        mask = np.ones(target.shape[:1] + (1,) + target.shape[1:3], dtype=np.float32)
        for index in keyframe_indices(len(drawings), keyframe_stride):
            control[index] = target[index]
            mask[index] = 0.0
    elif task == "character_color":
        target = np.stack([cache.rgb(source["color_paths"][frame_id]) for frame_id in drawings])
        control = np.stack([cache.rgb(source["sketch_paths"][frame_id]) for frame_id in drawings])
        mask = np.ones(target.shape[:1] + (1,) + target.shape[1:3], dtype=np.float32)
        # A colored drawing from outside the clip acts as the color model sheet.
        outside = [frame_id for frame_id in source["drawing_ids"] if frame_id not in set(drawings)]
        pool = outside or source["drawing_ids"]
        reference_id = pool[len(pool) // 2] if center else rng.choice(pool)
        reference = reference_image(source["color_paths"][reference_id], height=height, width=width)
    elif task == "compose_refine":
        target = np.stack([cache.rgb(source["composition_paths"][frame_id]) for frame_id in drawings])
        frames = []
        for frame_id in drawings:
            layer = cache.rgba(source["color_paths"][frame_id]).astype(np.float32)
            background = cache.rgb(source["background_paths"][frame_id]).astype(np.float32)
            alpha = layer[..., 3:4] / 255.0
            frames.append((layer[..., :3] * alpha + background * (1.0 - alpha)).round().astype(np.uint8))
        control = np.stack(frames)
        mask = np.ones(target.shape[:1] + (1,) + target.shape[1:3], dtype=np.float32)
        if compose_plate_reference and source["plate_static"] and source["plate_path"]:
            reference = reference_image(source["plate_path"], height=height, width=width)
    else:
        raise ValueError(f"unknown task: {task}")
    return {
        "target": torch.from_numpy(target).permute(0, 3, 1, 2).contiguous(),
        "control": torch.from_numpy(control).permute(0, 3, 1, 2).contiguous(),
        "mask": torch.from_numpy(mask),
        "reference": reference,
        "drawings": drawings,
    }


def prompt_for(task: str, caption: str) -> str:
    return f"{TASK_PROMPTS[task]} {caption}".strip()


class VaceAnitaDataset(Dataset):
    def __init__(
        self,
        sources: list[dict[str, Any]],
        *,
        captions: dict[str, list[tuple[set[str], str]]],
        height: int,
        width: int,
        max_frames: int,
        timeline_stride: int,
        keyframe_stride: int,
        compose_plate_reference: bool,
        hflip_prob: float,
        text_drop_prob: float,
        deterministic: bool = False,
    ) -> None:
        self.records = sources
        self.captions = captions
        self.height = height
        self.width = width
        self.max_frames = max_frames
        self.timeline_stride = timeline_stride
        self.keyframe_stride = keyframe_stride
        self.compose_plate_reference = compose_plate_reference
        self.hflip_prob = hflip_prob
        self.text_drop_prob = text_drop_prob
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.records[index]
        rng = random.Random(index) if self.deterministic else random.Random()
        sample = build_sample(
            source,
            height=self.height,
            width=self.width,
            max_frames=self.max_frames,
            timeline_stride=self.timeline_stride,
            keyframe_stride=self.keyframe_stride,
            compose_plate_reference=self.compose_plate_reference,
            rng=rng,
            center=self.deterministic,
        )
        reference = sample["reference"]
        reference_tensor = (
            letterbox_like_vace(reference, height=self.height, width=self.width) if reference is not None else None
        )
        video = sample["target"].float() / 127.5 - 1.0
        control = sample["control"].float() / 127.5 - 1.0
        mask = sample["mask"]
        if not self.deterministic and rng.random() < self.hflip_prob:
            video, control, mask = video.flip(-1), control.flip(-1), mask.flip(-1)
            if reference_tensor is not None:
                reference_tensor = reference_tensor.flip(-1)
        caption = caption_for(self.captions, source["key"], sample["drawings"])
        if not self.deterministic and rng.random() < self.text_drop_prob:
            caption = ""
        return {
            "video": video.permute(1, 0, 2, 3).contiguous(),
            "control": control.permute(1, 0, 2, 3).contiguous(),
            "mask": mask.permute(1, 0, 2, 3).contiguous(),
            "reference": reference_tensor,
            "prompt": prompt_for(source["task"], caption),
            "task": source["task"],
            "key": source["key"],
        }


def collate_single(batch: list[dict[str, Any]]) -> dict[str, Any]:
    # Clip length varies per shot, so the trainer runs batch size 1 and accumulates gradients.
    if len(batch) != 1:
        raise ValueError("VACE Anita training uses train_batch_size 1; raise gradient_accumulation_steps instead")
    return batch[0]


def latent_weight_map(
    video: torch.Tensor,
    control: torch.Tensor,
    *,
    num_reference: int,
    latent_height: int,
    latent_width: int,
    floor: float,
    threshold: float,
) -> torch.Tensor:
    """Loss weights that emphasise pixels the model must change (target differs from control)."""
    changed = ((video - control).abs().amax(dim=0) > threshold).float()  # (T, H, W)
    changed = F.adaptive_max_pool2d(changed[:, None], (latent_height, latent_width))[:, 0]
    # Wan VAE: latent frame 0 <- pixel frame 0, latent frame j <- pixel frames 4j-3..4j.
    groups = [changed[:1].amax(dim=0)] + [
        changed[start : start + 4].amax(dim=0) for start in range(1, changed.shape[0], 4)
    ]
    changed = torch.stack(groups)
    changed = F.max_pool2d(changed[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
    weights = floor + (1.0 - floor) * changed
    if num_reference:
        weights = torch.cat([torch.full_like(weights[:1], floor).repeat(num_reference, 1, 1), weights])
    return weights[None, None]


def vace_conditioning(
    pipe: Any,
    *,
    control: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    references = [[reference.to(device)]] if reference is not None else [[]]
    control = control[None].to(device)
    mask = mask[None].to(device)
    latents = pipe.prepare_video_latents(control, mask, references, device=device)
    masks = pipe.prepare_masks(mask, references)
    return torch.cat([latents, masks.to(latents)], dim=1).to(dtype), len(references[0])


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Train a Wan2.1 VACE LoRA on AnitaDataset production stages.")
    parser.add_argument("--config")
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="/data/shasegawa/t2a/models/Wan2.1-VACE-1.3B-diffusers",
    )
    parser.add_argument("--anita-root", default="/data/shasegawa/t2a/datasets/anita")
    parser.add_argument("--captions", default="/data/shasegawa/t2a/manifests/anita_production_captioned.jsonl")
    parser.add_argument("--plates-index", default="/data/shasegawa/t2a/manifests/anita_background_plates.jsonl")
    parser.add_argument("--output-dir", default="/data/shasegawa/t2a/outputs/wan21-vace-anita-lora")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--task-weights", help="Optional mixture, e.g. inbetween=1,character_color=1,compose_refine=1.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max-frames", type=int, default=33)
    parser.add_argument("--min-frames", type=int, default=17)
    parser.add_argument(
        "--timeline-stride",
        type=int,
        default=2,
        help=f"Sample every Nth frame of the {TIMELINE_FPS} fps timing-sheet timeline (2 -> 12 fps).",
    )
    parser.add_argument("--keyframe-stride", type=int, default=8)
    parser.add_argument("--compose-plate-reference", type=str_to_bool, default=True)
    parser.add_argument("--val-scenes", default="", help="Comma-separated work/scene or scene names to hold out.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-validation-samples", type=int, default=24)
    parser.add_argument("--validation-steps", type=int, default=250)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--text-drop-prob", type=float, default=0.0)
    parser.add_argument("--loss-floor", type=float, default=0.2, help="Loss weight for unchanged regions.")
    parser.add_argument("--change-threshold", type=float, default=0.08, help="Pixel change in [-1,1] units.")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=3000)
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    # Matches attention/FFN projections in both the main Wan blocks and the VACE control blocks.
    parser.add_argument("--target-modules", default="to_q,to_k,to_v,to_out.0,ffn.net.0.proj,ffn.net.2")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gradient-checkpointing", type=str_to_bool, default=True)
    parser.add_argument("--enable-vae-tiling", type=str_to_bool, default=False)
    parser.add_argument(
        "--vae-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
        help="bf16 encodes ~1.7x faster; the render script uses the same dtype so conditioning matches.",
    )
    parser.set_defaults(**defaults)
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args()
    return build_parser(load_config(config_args.config)).parse_args()


def load_split(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    sources = build_clip_sources(
        args.anita_root,
        tasks=tasks,
        plates_index=args.plates_index,
        min_frames=args.min_frames,
        timeline_stride=args.timeline_stride,
    )
    val_scenes = tuple(item.strip() for item in args.val_scenes.split(",") if item.strip())
    return split_sources(sources, val_scenes=val_scenes, val_ratio=args.val_ratio, seed=args.seed)


def dataset_from_args(
    args: argparse.Namespace,
    sources: list[dict[str, Any]],
    captions: dict[str, list[tuple[set[str], str]]],
    *,
    deterministic: bool,
) -> VaceAnitaDataset:
    return VaceAnitaDataset(
        sources,
        captions=captions,
        height=args.height,
        width=args.width,
        max_frames=args.max_frames,
        timeline_stride=args.timeline_stride,
        keyframe_stride=args.keyframe_stride,
        compose_plate_reference=args.compose_plate_reference,
        hflip_prob=args.hflip_prob,
        text_drop_prob=args.text_drop_prob,
        deterministic=deterministic,
    )


class PromptEncoder:
    def __init__(self, pipe: Any, device: torch.device, dtype: torch.dtype, max_length: int) -> None:
        self.pipe = pipe
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def __call__(self, prompt: str) -> torch.Tensor:
        if prompt not in self.cache:
            self.cache[prompt] = encode_prompt_like_wan(
                self.pipe.tokenizer, self.pipe.text_encoder, [prompt], self.device, self.dtype, self.max_length
            ).cpu()
        return self.cache[prompt].to(self.device)


def diffusion_loss(
    *,
    pipe: Any,
    transformer: torch.nn.Module,
    vae: torch.nn.Module,
    item: dict[str, Any],
    prompt_embeds: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    noise_generator: torch.Generator | None,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with torch.no_grad():
        control_states, num_reference = vace_conditioning(
            pipe,
            control=item["control"],
            mask=item["mask"],
            reference=item["reference"],
            device=device,
            dtype=dtype,
        )
        video = item["video"][None].to(device, vae.dtype)
        latents = normalize_wan_latents(vae, retrieve_latents(vae.encode(video), sample_mode="mean"))
        if num_reference:
            # Reference frames are prepended to the denoised sequence exactly as WanVACEPipeline does;
            # their clean target is the reference latent stored in the conditioning's inactive half.
            latents = torch.cat([control_states[:, :16, :num_reference].float(), latents], dim=2)
        weights = latent_weight_map(
            item["video"],
            item["control"],
            num_reference=num_reference,
            latent_height=latents.shape[-2],
            latent_width=latents.shape[-1],
            floor=args.loss_floor,
            threshold=args.change_threshold,
        ).to(device)
    noise = torch.randn(latents.shape, generator=noise_generator, device=device, dtype=latents.dtype)
    noisy = (1.0 - sigmas) * latents + sigmas * noise
    prediction = transformer(
        hidden_states=noisy.to(dtype),
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        control_hidden_states=control_states,
        return_dict=False,
    )[0]
    squared = (prediction.float() - (noise - latents).float()) ** 2
    return (squared * weights).sum() / (weights.sum() * squared.shape[1])


@torch.no_grad()
def run_validation(
    *,
    pipe: Any,
    transformer: torch.nn.Module,
    vae: torch.nn.Module,
    dataset: VaceAnitaDataset,
    encode_prompt: PromptEncoder,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    # Fixed windows, noise, and timesteps so losses are comparable across checkpoints.
    scheduler_timesteps = pipe.scheduler.timesteps.to(device)
    scheduler_sigmas = pipe.scheduler.sigmas.to(device)
    probes = [int(torch.argmin((scheduler_timesteps - value).abs())) for value in (250.0, 500.0, 750.0)]
    was_training = transformer.training
    transformer.eval()
    by_task: dict[str, list[float]] = {}
    for index in range(min(len(dataset), args.max_validation_samples)):
        item = dataset[index]
        prompt_embeds = encode_prompt(item["prompt"])
        losses = []
        for probe in probes:
            generator = torch.Generator(device=device).manual_seed(args.seed + index * 17 + probe)
            losses.append(
                diffusion_loss(
                    pipe=pipe,
                    transformer=transformer,
                    vae=vae,
                    item=item,
                    prompt_embeds=prompt_embeds,
                    timesteps=scheduler_timesteps[probe : probe + 1],
                    sigmas=scheduler_sigmas[probe].view(1, 1, 1, 1, 1),
                    noise_generator=generator,
                    args=args,
                    device=device,
                    dtype=dtype,
                ).item()
            )
        by_task.setdefault(item["task"], []).append(float(np.mean(losses)))
    if was_training:
        transformer.train()
    summary = {task: float(np.mean(values)) for task, values in by_task.items()}
    summary["mean"] = float(np.mean([value for values in by_task.values() for value in values])) if by_task else 0.0
    return summary


def main() -> None:
    args = parse_args()
    # Direct GPU peer copies silently return zeros on this host; keep NCCL off the P2P transport.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.optimization import get_scheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    if args.train_batch_size != 1:
        raise ValueError("train_batch_size must be 1 (clip length varies); use gradient_accumulation_steps")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    set_seed(args.seed)
    dtype = weight_dtype(args.mixed_precision)
    device = accelerator.device

    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=getattr(torch, args.vae_dtype)
    )
    pipe = WanVACEPipeline.from_pretrained(args.pretrained_model_name_or_path, vae=vae, torch_dtype=dtype)
    pipe.scheduler.set_timesteps(getattr(pipe.scheduler.config, "num_train_timesteps", 1000), device=device)
    vae = pipe.vae.to(device)
    if args.enable_vae_tiling:
        vae.enable_tiling()
    pipe.text_encoder.to(device)
    transformer = pipe.transformer.to(device)
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=[module.strip() for module in args.target_modules.split(",") if module.strip()],
        )
    )
    transformer.train()

    captions = load_captions(args.captions)
    train_sources, val_sources = load_split(args)
    if not train_sources:
        raise ValueError("no training clips found; check --anita-root, --plates-index, and --min-frames")
    tasks = tuple(sorted({source["task"] for source in train_sources}, key=TASKS.index))
    train_dataset = dataset_from_args(args, train_sources, captions, deterministic=False)
    val_dataset = dataset_from_args(args, val_sources, captions, deterministic=True)
    num_samples = args.max_train_steps * args.gradient_accumulation_steps * accelerator.num_processes + 1000
    sampler = WeightedRandomSampler(
        build_task_sample_weights(train_sources, tasks=tasks, task_weights=parse_task_weights(args.task_weights, TASKS)),
        num_samples=num_samples,
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_single,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    trainable = [param for param in transformer.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.adam_weight_decay)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    transformer, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_loader, lr_scheduler
    )
    encode_prompt = PromptEncoder(pipe, device, dtype, args.max_sequence_length)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
        split = {
            name: sorted({f"{source['key']}:{source['task']}" for source in sources})
            for name, sources in (("train", train_sources), ("val", val_sources))
        }
        (output_dir / "split.json").write_text(json.dumps(split, indent=2))
        counts = {task: sum(source["task"] == task for source in train_sources) for task in tasks}
        print(f"train clips per task: {counts}; validation clips: {len(val_sources)}")
    metadata = {
        "base_model": args.pretrained_model_name_or_path,
        "dataset": "AnitaDataset",
        "task": "vace-production-stages",
        "tasks": ",".join(tasks),
        "resolution": f"{args.width}x{args.height}",
        "max_frames": str(args.max_frames),
        "fps": str(TIMELINE_FPS // args.timeline_stride),
    }

    def save(directory: Path) -> None:
        save_lora(
            accelerator=accelerator,
            pipeline_cls=WanVACEPipeline,
            transformer=transformer,
            output_dir=directory,
            convert_state_dict_to_diffusers=convert_state_dict_to_diffusers,
            get_peft_model_state_dict=get_peft_model_state_dict,
            metadata=metadata,
        )

    def validate(step: int) -> None:
        if accelerator.is_main_process and len(val_dataset):
            summary = run_validation(
                pipe=pipe,
                transformer=accelerator.unwrap_model(transformer),
                vae=vae,
                dataset=val_dataset,
                encode_prompt=encode_prompt,
                args=args,
                device=device,
                dtype=dtype,
            )
            with (output_dir / "validation_log.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **summary}) + "\n")
            print(f"step {step} validation: {json.dumps(summary)}")
        accelerator.wait_for_everyone()

    validate(0)
    global_step = 0
    running: dict[str, list[float]] = {}
    progress = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process, desc="wan-vace-anita")
    for item in train_loader:
        with accelerator.accumulate(transformer):
            timesteps, sigmas = sample_flow_timesteps(
                pipe.scheduler, 1, device, train_stage="high", boundary_ratio=None
            )
            loss = diffusion_loss(
                pipe=pipe,
                transformer=transformer,
                vae=vae,
                item=item,
                prompt_embeds=encode_prompt(item["prompt"]),
                timesteps=timesteps,
                sigmas=sigmas,
                noise_generator=None,
                args=args,
                device=device,
                dtype=dtype,
            )
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        running.setdefault(item["task"], []).append(loss.detach().item())

        if accelerator.sync_gradients:
            global_step += 1
            progress.update(1)
            if accelerator.is_main_process and global_step % 10 == 0:
                means = {task: float(np.mean(values[-50:])) for task, values in running.items()}
                progress.set_postfix({task[:5]: f"{value:.4f}" for task, value in means.items()})
                with (output_dir / "train_log.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps({"step": global_step, "lr": lr_scheduler.get_last_lr()[0], **means}) + "\n"
                    )
            if global_step % args.checkpointing_steps == 0:
                save(output_dir / f"checkpoint-{global_step}")
                accelerator.wait_for_everyone()
            if global_step % args.validation_steps == 0:
                validate(global_step)
            if global_step >= args.max_train_steps:
                break

    save(output_dir)
    if global_step % args.validation_steps != 0:
        validate(global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
