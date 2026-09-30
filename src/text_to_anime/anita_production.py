from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from functools import lru_cache
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

from .manifest import read_jsonl, write_jsonl
from .train_wan_lora import load_config, str_to_bool, weight_dtype
from .video import resize_and_crop, sample_indices

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TASKS = ("line_art", "character_color", "compose_refine")
SHOT_SPLIT_STRATEGIES = ("semantic", "frame_count")
CONDITION_CHANNELS = 12


@dataclass(frozen=True)
class AnitaShotSet:
    work: str
    scene: str
    sketch_paths: dict[str, str]
    color_paths: dict[str, str]
    composition_paths: dict[str, str]


@dataclass(frozen=True)
class AnitaProductionShot:
    task: str
    prompt: str
    primary_paths: list[str]
    target_paths: list[str]
    reference_paths: list[str]
    mask_paths: list[str]
    work: str
    scene: str
    frame_ids: list[str]
    parent_scene: str | None = None
    subshot_index: int = 0
    num_subshots: int = 1
    split_strategy: str = "none"


def _image_files(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {
        path.stem: str(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def _has_images(directory: Path) -> bool:
    return bool(_image_files(directory))


def iter_shot_sets(root: str | Path) -> Iterable[AnitaShotSet]:
    root = Path(root)
    emitted: set[tuple[str, str]] = set()

    # Layout used by the extracted Anita archive: work/{sketch,color,composition}/scene/*.png
    for work_dir in sorted(path for path in root.rglob("*") if path.is_dir()):
        stage_dirs = {stage: work_dir / stage for stage in ("sketch", "color", "composition")}
        if not any(stage_dir.is_dir() for stage_dir in stage_dirs.values()):
            continue
        scene_names: set[str] = set()
        for stage_dir in stage_dirs.values():
            if not stage_dir.is_dir():
                continue
            scene_names.update(
                scene_dir.name
                for scene_dir in stage_dir.iterdir()
                if scene_dir.is_dir() and _has_images(scene_dir)
            )
        relative = work_dir.relative_to(root)
        work = "/".join(relative.parts) if relative.parts else "AnitaDataset"
        for scene_name in sorted(scene_names):
            key = (work, scene_name)
            emitted.add(key)
            yield AnitaShotSet(
                work=work,
                scene=scene_name,
                sketch_paths=_image_files(stage_dirs["sketch"] / scene_name),
                color_paths=_image_files(stage_dirs["color"] / scene_name),
                composition_paths=_image_files(stage_dirs["composition"] / scene_name),
            )

    # Also support work/scene/{sketch,color,composition}/*.png for synthetic tests/custom data.
    for scene_dir in sorted(path for path in root.rglob("*") if path.is_dir()):
        sketch_dir = scene_dir / "sketch"
        color_dir = scene_dir / "color"
        composition_dir = scene_dir / "composition"
        if not any(_has_images(stage_dir) for stage_dir in (sketch_dir, color_dir, composition_dir)):
            continue
        relative = scene_dir.relative_to(root)
        work = relative.parts[0] if relative.parts else "AnitaDataset"
        scene = "/".join(relative.parts[1:]) if len(relative.parts) > 1 else scene_dir.name
        key = (work, scene)
        if key in emitted:
            continue
        yield AnitaShotSet(
            work=work,
            scene=scene,
            sketch_paths=_image_files(sketch_dir),
            color_paths=_image_files(color_dir),
            composition_paths=_image_files(composition_dir),
        )


def _paths_for(frame_map: dict[str, str], frame_ids: Iterable[str]) -> list[str]:
    return [frame_map[frame_id] for frame_id in frame_ids]


def _target_chunk_count(
    *,
    total: int,
    max_frames_per_shot: int | None,
    min_frames_per_shot: int,
) -> int:
    if not max_frames_per_shot or max_frames_per_shot <= 0:
        return 1
    if total <= max_frames_per_shot:
        return 1
    min_frames_per_shot = max(1, min_frames_per_shot)
    chunk_count = math.ceil(total / max_frames_per_shot)
    while chunk_count > 1 and total // chunk_count < min_frames_per_shot:
        chunk_count -= 1
    return chunk_count


def _balanced_split_frame_ids(
    frame_ids: list[str],
    *,
    max_frames_per_shot: int | None,
    min_frames_per_shot: int,
) -> list[list[str]]:
    chunk_count = _target_chunk_count(
        total=len(frame_ids),
        max_frames_per_shot=max_frames_per_shot,
        min_frames_per_shot=min_frames_per_shot,
    )
    if chunk_count <= 1:
        return [frame_ids]

    base_size, remainder = divmod(len(frame_ids), chunk_count)
    chunks: list[list[str]] = []
    start = 0
    for index in range(chunk_count):
        size = base_size + (1 if index < remainder else 0)
        chunks.append(frame_ids[start : start + size])
        start += size
    return chunks


@lru_cache(maxsize=65536)
def _semantic_frame_feature(path: str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    gray_image = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8))
    gray_small = np.asarray(
        gray_image.resize((16, 16), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    histograms = []
    for channel in range(3):
        histogram, _ = np.histogram(rgb[..., channel], bins=16, range=(0.0, 1.0))
        histogram = histogram.astype(np.float32)
        histogram /= max(float(histogram.sum()), 1.0)
        histograms.append(histogram)
    return np.concatenate([gray_small.reshape(-1), *histograms])


def _semantic_transition_score(
    frame_ids: list[str],
    frame_paths: dict[str, str],
    cut: int,
) -> float | None:
    if cut <= 0 or cut >= len(frame_ids):
        return None
    try:
        previous = _semantic_frame_feature(frame_paths[frame_ids[cut - 1]])
        current = _semantic_frame_feature(frame_paths[frame_ids[cut]])
    except (FileNotFoundError, KeyError, OSError):
        return None
    structure_delta = np.mean(np.abs(previous[:256] - current[:256]))
    color_delta = np.mean(np.abs(previous[256:] - current[256:]))
    return float(structure_delta + 0.5 * color_delta)


def _semantic_cut_points(
    *,
    frame_ids: list[str],
    frame_paths: dict[str, str],
    chunk_count: int,
    max_frames_per_shot: int,
    min_frames_per_shot: int,
    search_radius: int,
) -> list[int]:
    total = len(frame_ids)
    min_frames_per_shot = max(1, min_frames_per_shot)
    search_radius = max(1, search_radius)
    cut_points: list[int] = []
    previous_cut = 0
    for boundary_index in range(1, chunk_count):
        remaining_chunks = chunk_count - boundary_index
        lower = max(
            previous_cut + min_frames_per_shot,
            total - remaining_chunks * max_frames_per_shot,
        )
        upper = min(
            previous_cut + max_frames_per_shot,
            total - remaining_chunks * min_frames_per_shot,
        )
        if lower > upper:
            return []

        ideal = round(total * boundary_index / chunk_count)
        candidate_lower = max(lower, ideal - search_radius)
        candidate_upper = min(upper, ideal + search_radius)
        scored_candidates = []
        for cut in range(candidate_lower, candidate_upper + 1):
            score = _semantic_transition_score(frame_ids, frame_paths, cut)
            if score is not None:
                scored_candidates.append((cut, score))
        if not scored_candidates:
            return []

        raw_scores = np.asarray([score for _, score in scored_candidates], dtype=np.float32)
        spread = float(raw_scores.max() - raw_scores.min())
        if spread < 1e-6:
            return []
        score_min = float(raw_scores.min())
        best_cut = scored_candidates[0][0]
        best_score = -float("inf")
        for cut, raw_score in scored_candidates:
            dynamic_score = (raw_score - score_min) / spread
            proximity_penalty = abs(cut - ideal) / max(1.0, float(search_radius))
            score = dynamic_score - 0.2 * proximity_penalty
            if score > best_score:
                best_cut = cut
                best_score = score
        cut_points.append(best_cut)
        previous_cut = best_cut
    return cut_points


def _split_by_cut_points(frame_ids: list[str], cut_points: list[int]) -> list[list[str]]:
    chunks = []
    start = 0
    for cut in cut_points:
        chunks.append(frame_ids[start:cut])
        start = cut
    chunks.append(frame_ids[start:])
    return chunks


def split_frame_ids(
    frame_ids: list[str],
    *,
    max_frames_per_shot: int | None,
    min_frames_per_shot: int,
    frame_paths: dict[str, str] | None = None,
    split_strategy: str = "semantic",
    semantic_search_radius: int = 3,
) -> list[list[str]]:
    chunk_count = _target_chunk_count(
        total=len(frame_ids),
        max_frames_per_shot=max_frames_per_shot,
        min_frames_per_shot=min_frames_per_shot,
    )
    if chunk_count <= 1:
        return [frame_ids]
    if split_strategy == "frame_count" or frame_paths is None or max_frames_per_shot is None:
        return _balanced_split_frame_ids(
            frame_ids,
            max_frames_per_shot=max_frames_per_shot,
            min_frames_per_shot=min_frames_per_shot,
        )
    if split_strategy != "semantic":
        raise ValueError(f"unknown shot split strategy: {split_strategy}")

    cut_points = _semantic_cut_points(
        frame_ids=frame_ids,
        frame_paths=frame_paths,
        chunk_count=chunk_count,
        max_frames_per_shot=max_frames_per_shot,
        min_frames_per_shot=min_frames_per_shot,
        search_radius=semantic_search_radius,
    )
    if not cut_points:
        return _balanced_split_frame_ids(
            frame_ids,
            max_frames_per_shot=max_frames_per_shot,
            min_frames_per_shot=min_frames_per_shot,
        )
    return _split_by_cut_points(frame_ids, cut_points)


def _subshot_scene(scene: str, index: int, total: int) -> str:
    if total <= 1:
        return scene
    return f"{scene}_part{index:03d}"


def _chunk_prompt(base_prompt: str, frame_ids: list[str], index: int, total: int) -> str:
    if total <= 1 or not frame_ids:
        return base_prompt
    return (
        f"{base_prompt} Sub-shot {index + 1} of {total}, frames "
        f"{frame_ids[0]} through {frame_ids[-1]}."
    )


def _prompt_for(shot: AnitaShotSet, task: str) -> str:
    if task == "line_art":
        return (
            f"Generate a continuous anime production shot of clean line-art frames for "
            f"{shot.work}, {shot.scene}."
        )
    if task == "character_color":
        return (
            f"Colorize the main animated characters across the line-art shot for "
            f"{shot.work}, {shot.scene}, while leaving the background unfinished."
        )
    if task == "compose_refine":
        return (
            f"Refine the character-colored shot and complete the background coloring and "
            f"compositing for {shot.work}, {shot.scene}."
        )
    raise ValueError(f"unknown task: {task}")


def build_production_samples(
    root: str | Path,
    *,
    tasks: Iterable[str] = TASKS,
    max_frames_per_shot: int | None = None,
    min_frames_per_shot: int = 8,
    shot_split_strategy: str = "semantic",
    semantic_split_search_radius: int = 3,
) -> list[dict[str, Any]]:
    enabled = set(tasks)
    samples: list[AnitaProductionShot] = []

    def add_chunks(
        *,
        shot: AnitaShotSet,
        task: str,
        frame_ids: list[str],
        primary_map: dict[str, str] | None,
        target_map: dict[str, str],
        reference_map: dict[str, str],
        mask_map: dict[str, str] | None,
    ) -> None:
        chunks = split_frame_ids(
            frame_ids,
            max_frames_per_shot=max_frames_per_shot,
            min_frames_per_shot=min_frames_per_shot,
            frame_paths=target_map,
            split_strategy=shot_split_strategy,
            semantic_search_radius=semantic_split_search_radius,
        )
        applied_split_strategy = "none"
        if len(chunks) > 1:
            applied_split_strategy = shot_split_strategy
            if shot_split_strategy == "semantic":
                balanced_chunks = _balanced_split_frame_ids(
                    frame_ids,
                    max_frames_per_shot=max_frames_per_shot,
                    min_frames_per_shot=min_frames_per_shot,
                )
                if chunks == balanced_chunks:
                    applied_split_strategy = "frame_count"
        base_prompt = _prompt_for(shot, task)
        reference_ids = sorted(reference_map)
        for chunk_index, chunk_ids in enumerate(chunks):
            samples.append(
                AnitaProductionShot(
                    task=task,
                    prompt=_chunk_prompt(base_prompt, chunk_ids, chunk_index, len(chunks)),
                    primary_paths=_paths_for(primary_map, chunk_ids) if primary_map else [],
                    target_paths=_paths_for(target_map, chunk_ids),
                    reference_paths=_paths_for(reference_map, reference_ids),
                    mask_paths=_paths_for(mask_map, chunk_ids) if mask_map else [],
                    work=shot.work,
                    scene=_subshot_scene(shot.scene, chunk_index, len(chunks)),
                    frame_ids=chunk_ids,
                    parent_scene=shot.scene,
                    subshot_index=chunk_index,
                    num_subshots=len(chunks),
                    split_strategy=applied_split_strategy,
                )
            )

    for shot in iter_shot_sets(root):
        if "line_art" in enabled and shot.sketch_paths:
            add_chunks(
                shot=shot,
                task="line_art",
                frame_ids=sorted(shot.sketch_paths),
                primary_map=None,
                target_map=shot.sketch_paths,
                reference_map=shot.sketch_paths,
                mask_map=None,
            )

        color_ids = sorted(set(shot.sketch_paths) & set(shot.color_paths))
        if "character_color" in enabled and color_ids:
            add_chunks(
                shot=shot,
                task="character_color",
                frame_ids=color_ids,
                primary_map=shot.sketch_paths,
                target_map=shot.color_paths,
                reference_map=shot.color_paths,
                mask_map=shot.color_paths,
            )

        composition_ids = sorted(set(shot.color_paths) & set(shot.composition_paths))
        if "compose_refine" in enabled and composition_ids:
            add_chunks(
                shot=shot,
                task="compose_refine",
                frame_ids=composition_ids,
                primary_map=shot.color_paths,
                target_map=shot.composition_paths,
                reference_map=shot.composition_paths,
                mask_map=shot.color_paths,
            )
    return [asdict(sample) for sample in samples]


def _load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _pil_to_chw(image: Image.Image) -> torch.Tensor:
    data = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.uint8))
    return data.permute(2, 0, 1).contiguous()


def _alpha_or_nonwhite_mask(path: str | Path, *, threshold: int = 250) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    rgba = np.array(image, dtype=np.uint8)
    alpha = rgba[..., 3]
    if alpha.min() < 255:
        mask = alpha
    else:
        rgb = rgba[..., :3]
        mask = (
            np.abs(rgb.astype(np.int16) - 255).max(axis=-1) > (255 - threshold)
        ).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def _blank_rgb(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), (0, 0, 0))


def _blank_mask(width: int, height: int) -> Image.Image:
    return Image.new("L", (width, height), 0)


def _load_sequence(paths: list[str], indices: list[int]) -> torch.Tensor:
    frames = [_pil_to_chw(_load_rgb(paths[index])) for index in indices]
    return torch.stack(frames, dim=0)


def _load_mask_sequence(paths: list[str], indices: list[int]) -> torch.Tensor:
    frames = []
    for index in indices:
        mask = _alpha_or_nonwhite_mask(paths[index])
        data = torch.from_numpy(np.array(mask, dtype=np.uint8))[None]
        frames.append(data.contiguous())
    return torch.stack(frames, dim=0)


def _resize_channels(
    tensors: list[torch.Tensor],
    *,
    height: int,
    width: int,
    random_crop: bool,
) -> list[torch.Tensor]:
    stacked = torch.cat(tensors, dim=1)
    resized = resize_and_crop(stacked, height=height, width=width, random_crop=random_crop)
    out: list[torch.Tensor] = []
    offset = 0
    for tensor in tensors:
        count = tensor.shape[1]
        out.append(resized[:, offset : offset + count])
        offset += count
    return out


def _normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float() / 127.5 - 1.0


def _normalize_mask(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float() / 255.0


def _repeat_image(image: Image.Image, num_frames: int) -> torch.Tensor:
    tensor = _pil_to_chw(image)
    return tensor.unsqueeze(0).repeat(num_frames, 1, 1, 1)


def _choose_reference(
    paths: list[str],
    selected_indices: list[int],
    target_size: tuple[int, int],
) -> Image.Image:
    if not paths:
        return _blank_rgb(*target_size)
    candidates = list(range(len(paths)))
    if len(candidates) > 1:
        selected = set(selected_indices)
        candidates = [index for index in candidates if index not in selected] or candidates
    return _load_rgb(paths[random.choice(candidates)])


class AnitaProductionDataset(Dataset):
    def __init__(
        self,
        manifest_or_root: str | Path,
        *,
        num_frames: int,
        height: int,
        width: int,
        random_clip: bool,
        random_crop: bool,
        first_frame_prob: float,
        reference_prob: float,
        text_drop_prob: float,
        tasks: Iterable[str] = TASKS,
    ) -> None:
        path = Path(manifest_or_root)
        if path.is_file():
            records = read_jsonl(path)
        else:
            records = build_production_samples(path, tasks=tasks)
        enabled = set(tasks)
        self.records = [record for record in records if record.get("task") in enabled]
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.random_clip = random_clip
        self.random_crop = random_crop
        self.first_frame_prob = first_frame_prob
        self.reference_prob = reference_prob
        self.text_drop_prob = text_drop_prob

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target_paths = record["target_paths"]
        if not target_paths:
            raise ValueError(f"record has no target_paths: {record}")
        indices = sample_indices(len(target_paths), self.num_frames, random_clip=self.random_clip)
        target = _load_sequence(target_paths, indices)
        _, _, source_height, source_width = target.shape
        target_size = (source_width, source_height)

        primary_paths = record.get("primary_paths") or []
        if primary_paths:
            primary = _load_sequence(primary_paths, indices)
            primary_flag = torch.full((self.num_frames, 1, source_height, source_width), 255)
        else:
            primary = torch.zeros(
                self.num_frames, 3, source_height, source_width, dtype=torch.uint8
            )
            primary_flag = torch.zeros(
                self.num_frames, 1, source_height, source_width, dtype=torch.uint8
            )

        include_first = random.random() < self.first_frame_prob
        if include_first:
            first_image = _load_rgb(target_paths[indices[0]])
            first = _repeat_image(first_image, self.num_frames)
            first_flag = torch.full((self.num_frames, 1, source_height, source_width), 255)
        else:
            first = torch.zeros_like(primary)
            first_flag = torch.zeros_like(primary_flag)

        include_reference = random.random() < self.reference_prob
        if include_reference:
            ref_image = _choose_reference(record.get("reference_paths") or [], indices, target_size)
            reference = _repeat_image(ref_image, self.num_frames)
            ref_flag = torch.full((self.num_frames, 1, source_height, source_width), 255)
        else:
            reference = torch.zeros_like(primary)
            ref_flag = torch.zeros_like(primary_flag)

        mask_paths = record.get("mask_paths") or []
        if mask_paths:
            foreground_mask = _load_mask_sequence(mask_paths, indices)
        else:
            foreground_mask = torch.zeros_like(primary_flag)

        (
            primary,
            first,
            reference,
            primary_flag,
            first_flag,
            ref_flag,
            foreground_mask,
            target,
        ) = _resize_channels(
            [
                primary,
                first,
                reference,
                primary_flag,
                first_flag,
                ref_flag,
                foreground_mask,
                target,
            ],
            height=self.height,
            width=self.width,
            random_crop=self.random_crop,
        )
        condition = torch.cat(
            [
                _normalize_image(primary),
                _normalize_image(first),
                _normalize_image(reference),
                _normalize_mask(primary_flag),
                _normalize_mask(first_flag),
                _normalize_mask(ref_flag),
            ],
            dim=1,
        )
        prompt = str(record.get("prompt", "")).strip()
        caption = str(record.get("caption", "")).strip()
        if caption and caption not in prompt:
            prompt = f"{prompt} Shot content: {caption}".strip()
        if random.random() < self.text_drop_prob:
            prompt = ""
        return {
            "condition": condition.permute(1, 0, 2, 3).contiguous(),
            "target": _normalize_image(target).permute(1, 0, 2, 3).contiguous(),
            "mask": _normalize_mask(foreground_mask).permute(1, 0, 2, 3).contiguous(),
            "task": record["task"],
            "prompt": prompt,
            "record": record,
        }


def collate_production_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "condition": torch.stack([item["condition"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "task": [item["task"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "record": [item["record"] for item in batch],
    }


def _hash_token(token: str, vocab_size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % vocab_size


def tokenize_prompts(
    prompts: list[str],
    *,
    vocab_size: int,
    max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for prompt in prompts:
        tokens = [token.strip(".,;:!?()[]{}\"'").lower() for token in prompt.split()]
        ids = [_hash_token(token, vocab_size) for token in tokens if token][:max_tokens]
        if not ids:
            ids = [0]
        ids.extend([0] * (max_tokens - len(ids)))
        rows.append(ids)
    return torch.tensor(rows, device=device, dtype=torch.long)


class TextTaskConditioner(nn.Module):
    def __init__(
        self,
        *,
        task_names: Iterable[str] = TASKS,
        cond_dim: int = 256,
        vocab_size: int = 8192,
        max_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.task_to_id = {task: index for index, task in enumerate(task_names)}
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.text_embedding = nn.Embedding(vocab_size, cond_dim, padding_idx=0)
        self.task_embedding = nn.Embedding(len(self.task_to_id), cond_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

    def forward(self, tasks: list[str], prompts: list[str], device: torch.device) -> torch.Tensor:
        token_ids = tokenize_prompts(
            prompts,
            vocab_size=self.vocab_size,
            max_tokens=self.max_tokens,
            device=device,
        )
        text = self.text_embedding(token_ids)
        mask = token_ids.ne(0).unsqueeze(-1)
        text = (text * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        task_ids = torch.tensor(
            [self.task_to_id[task] for task in tasks],
            device=device,
            dtype=torch.long,
        )
        return self.proj(text + self.task_embedding(task_ids))


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConditionalResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.cond = nn.Linear(cond_dim, out_channels * 2)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(cond).chunk(2, dim=1)
        x = self.norm2(x) * (1 + scale[..., None, None, None])
        x = x + shift[..., None, None, None]
        x = self.conv2(F.silu(x))
        return x + residual


class DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.res = ConditionalResBlock3D(in_channels, out_channels, cond_dim)
        self.down = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.res(x, cond)
        return self.down(skip), skip


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.res = ConditionalResBlock3D(out_channels + skip_channels, out_channels, cond_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.res(torch.cat([x, skip], dim=1), cond)


class AnitaProductionUNet3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = CONDITION_CHANNELS,
        out_channels: int = 3,
        base_channels: int = 48,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        cond_dim: int = 256,
        vocab_size: int = 8192,
        max_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "channel_mults": list(channel_mults),
            "cond_dim": cond_dim,
            "vocab_size": vocab_size,
            "max_tokens": max_tokens,
            "tasks": list(TASKS),
            "condition_channels": CONDITION_CHANNELS,
            "model_type": "AnitaProductionUNet3D",
        }
        self.conditioner = TextTaskConditioner(
            task_names=TASKS,
            cond_dim=cond_dim,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
        )
        channels = [base_channels * mult for mult in channel_mults]
        self.input = nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList()
        previous = channels[0]
        for channel in channels:
            self.downs.append(DownBlock3D(previous, channel, cond_dim))
            previous = channel
        self.mid = nn.ModuleList(
            [
                ConditionalResBlock3D(previous, previous, cond_dim),
                ConditionalResBlock3D(previous, previous, cond_dim),
            ]
        )
        self.ups = nn.ModuleList()
        for skip_channel in reversed(channels):
            out_channel = skip_channel
            self.ups.append(UpBlock3D(previous, skip_channel, out_channel, cond_dim))
            previous = out_channel
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(previous), previous),
            nn.SiLU(),
            nn.Conv3d(previous, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        condition: torch.Tensor,
        tasks: list[str],
        prompts: list[str],
    ) -> torch.Tensor:
        cond = self.conditioner(tasks, prompts, condition.device)
        x = self.input(condition)
        skips = []
        for down in self.downs:
            x, skip = down(x, cond)
            skips.append(skip)
        for block in self.mid:
            x = block(x, cond)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, cond)
        return self.output(x)


AnitaProductionUNet = AnitaProductionUNet3D


def task_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    tasks: list[str],
    *,
    foreground_weight: float,
    sketch_edge_weight: float,
    temporal_weight: float,
) -> torch.Tensor:
    weights = torch.ones_like(mask)
    for index, task in enumerate(tasks):
        if task in {"character_color", "compose_refine"}:
            weights[index] = 1.0 + mask[index] * foreground_weight
    loss = ((prediction - target).abs() * weights).mean()

    if temporal_weight > 0 and prediction.shape[2] > 1:
        pred_delta = prediction[:, :, 1:] - prediction[:, :, :-1]
        target_delta = target[:, :, 1:] - target[:, :, :-1]
        loss = loss + temporal_weight * (pred_delta - target_delta).abs().mean()

    if sketch_edge_weight > 0 and any(task == "line_art" for task in tasks):
        sketch_indices = [index for index, task in enumerate(tasks) if task == "line_art"]
        pred_gray = prediction[sketch_indices].mean(dim=1, keepdim=True)
        target_gray = target[sketch_indices].mean(dim=1, keepdim=True)
        bsz, _, frames, height, width = pred_gray.shape
        pred_gray = pred_gray.transpose(1, 2).reshape(bsz * frames, 1, height, width)
        target_gray = target_gray.transpose(1, 2).reshape(bsz * frames, 1, height, width)
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, 1, 3, 3)
        edge_loss = (
            F.conv2d(pred_gray, kernel, padding=1) - F.conv2d(target_gray, kernel, padding=1)
        ).abs().mean()
        loss = loss + sketch_edge_weight * edge_loss
    return loss


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    pixels = ((tensor.detach().cpu().float().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


def _contact_sheet(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("cannot build a contact sheet with no images")
    sheet = Image.new("RGB", (images[0].width * len(images), images[0].height), (255, 255, 255))
    for index, image in enumerate(images):
        sheet.paste(image, (index * image.width, 0))
    return sheet


def save_sequence_frames(video: torch.Tensor, output: str | Path) -> None:
    output = Path(output)
    frames = [tensor_to_pil(video[:, index]) for index in range(video.shape[1])]
    if output.suffix.lower() in IMAGE_EXTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        _contact_sheet(frames).save(output)
        frame_dir = output.with_suffix("")
    else:
        frame_dir = output
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(frame_dir / f"{index:04d}.png")


def save_validation_grid(
    *,
    model: AnitaProductionUNet3D,
    batch: dict[str, Any],
    output_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model.eval()
    condition = batch["condition"].to(device=device, dtype=dtype)
    with torch.no_grad():
        prediction = model(condition, batch["task"], batch["prompt"]).float()
    rows = []
    for batch_index in range(min(condition.shape[0], 2)):
        frame_indices = sorted({0, condition.shape[2] // 2, condition.shape[2] - 1})
        images = []
        for frame_index in frame_indices:
            images.extend(
                [
                    tensor_to_pil(condition[batch_index, :3, frame_index].float()),
                    tensor_to_pil(condition[batch_index, 6:9, frame_index].float()),
                    tensor_to_pil(prediction[batch_index, :, frame_index]),
                    tensor_to_pil(batch["target"][batch_index, :, frame_index]),
                ]
            )
        rows.append(_contact_sheet(images))
    if not rows:
        return
    grid = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), (255, 255, 255))
    for row_index, row in enumerate(rows):
        grid.paste(row, (0, row_index * row.height))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    model.train()


def save_checkpoint(
    *,
    model: AnitaProductionUNet3D,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model.config,
            "args": vars(args),
        },
        output_dir / "checkpoint.pt",
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(model.config, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_production_model(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> AnitaProductionUNet3D:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    config = checkpoint["model_config"].copy()
    config["channel_mults"] = tuple(config["channel_mults"])
    config.pop("tasks", None)
    config.pop("condition_channels", None)
    config.pop("model_type", None)
    model = AnitaProductionUNet3D(**config)
    model.load_state_dict(checkpoint["model"])
    return model


def cmd_manifest(args: argparse.Namespace) -> int:
    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    records = build_production_samples(
        args.root,
        tasks=tasks,
        max_frames_per_shot=args.max_frames_per_shot,
        min_frames_per_shot=args.min_frames_per_shot,
        shot_split_strategy=args.shot_split_strategy,
        semantic_split_search_radius=args.semantic_split_search_radius,
    )
    write_jsonl(args.output, records)
    print(json.dumps({"records": len(records), "output": args.output}, indent=2, sort_keys=True))
    return 0


def build_train_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(
        description="Train the all-in-one Anita shot production model."
    )
    parser.set_defaults(**defaults)
    parser.add_argument("--config")
    parser.add_argument(
        "--data",
        required="data" not in defaults,
        help="Anita root or production-shot JSONL manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/shasegawa/t2a/outputs/anita-production-unet3d",
    )
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--channel-mults", default="1,2,4")
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--foreground-weight", type=float, default=2.0)
    parser.add_argument("--sketch-edge-weight", type=float, default=0.05)
    parser.add_argument("--temporal-weight", type=float, default=0.1)
    parser.add_argument("--first-frame-prob", type=float, default=0.5)
    parser.add_argument("--reference-prob", type=float, default=0.5)
    parser.add_argument("--text-drop-prob", type=float, default=0.1)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--validation-steps", type=int, default=500)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--random-clip", type=str_to_bool, default=True)
    parser.add_argument("--random-crop", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args(argv)
    config = load_config(config_args.config)
    return build_train_parser(config).parse_args(argv)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = weight_dtype(args.mixed_precision) if device.type == "cuda" else torch.float32
    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    channel_mults = tuple(
        int(item.strip()) for item in args.channel_mults.split(",") if item.strip()
    )

    dataset = AnitaProductionDataset(
        args.data,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        random_clip=args.random_clip,
        random_crop=args.random_crop,
        first_frame_prob=args.first_frame_prob,
        reference_prob=args.reference_prob,
        text_drop_prob=args.text_drop_prob,
        tasks=tasks,
    )
    if not dataset:
        raise ValueError(f"no Anita production-shot samples found in {args.data}")
    val_size = max(1, int(len(dataset) * args.validation_ratio)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if val_size:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_production_batch,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_production_batch,
        )
        if val_dataset is not None
        else None
    )
    val_iter = iter(val_loader) if val_loader is not None else None

    model = AnitaProductionUNet3D(
        base_channels=args.base_channels,
        channel_mults=channel_mults,
        cond_dim=args.cond_dim,
        vocab_size=args.vocab_size,
        max_tokens=args.max_tokens,
    ).to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.mixed_precision == "fp16",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )

    global_step = 0
    num_epochs = math.ceil(args.max_train_steps / max(len(train_loader), 1))
    progress = tqdm(total=args.max_train_steps, desc="anita-production-shot")
    for _epoch in range(num_epochs):
        for batch in train_loader:
            model.train()
            condition = batch["condition"].to(device=device, dtype=dtype)
            target = batch["target"].to(device=device, dtype=dtype)
            mask = batch["mask"].to(device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda" and dtype != torch.float32,
            ):
                prediction = model(condition, batch["task"], batch["prompt"])
                loss = task_loss(
                    prediction,
                    target,
                    mask,
                    batch["task"],
                    foreground_weight=args.foreground_weight,
                    sketch_edge_weight=args.sketch_edge_weight,
                    temporal_weight=args.temporal_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.detach().float().item():.4f}")

            if global_step % args.checkpointing_steps == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    args=args,
                    output_dir=output_dir / f"checkpoint-{global_step}",
                )
            if (
                val_iter is not None
                and args.validation_steps
                and global_step % args.validation_steps == 0
            ):
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)
                save_validation_grid(
                    model=model,
                    batch=val_batch,
                    output_path=output_dir / "validation" / f"step_{global_step:06d}.jpg",
                    device=device,
                    dtype=dtype,
                )
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=global_step,
        args=args,
        output_dir=output_dir,
    )


def cmd_train(args: argparse.Namespace) -> int:
    train(args)
    return 0


def train_main() -> None:
    args = parse_train_args()
    train(args)


def _paths_from_input(path: str | None) -> list[str]:
    if not path:
        return []
    input_path = Path(path)
    if input_path.is_dir():
        return [
            str(frame)
            for frame in sorted(input_path.iterdir())
            if frame.is_file() and frame.suffix.lower() in IMAGE_EXTS
        ]
    return [str(input_path)]


def _load_condition_sequence(
    paths: list[str],
    *,
    num_frames: int,
    target_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    width, height = target_size
    if not paths:
        frames = torch.zeros(num_frames, 3, height, width, dtype=torch.uint8)
        flag = torch.zeros(num_frames, 1, height, width, dtype=torch.uint8)
        return frames, flag
    if len(paths) == 1:
        image = _load_rgb(paths[0]).resize(target_size, Image.Resampling.BICUBIC)
        frames = _repeat_image(image, num_frames)
    else:
        indices = sample_indices(len(paths), num_frames, random_clip=False)
        frames = _load_sequence(paths, indices)
        height_target, width_target = height, width
        if frames.shape[-2:] != (height_target, width_target):
            frames = resize_and_crop(
                frames,
                height=height_target,
                width=width_target,
                random_crop=False,
            )
    flag = torch.full((num_frames, 1, frames.shape[-2], frames.shape[-1]), 255, dtype=torch.uint8)
    return frames, flag


def _infer_target_size(args: argparse.Namespace) -> tuple[int, int]:
    for path in (args.source, args.first_frame, args.reference, args.aux):
        paths = _paths_from_input(path)
        if paths:
            return Image.open(paths[0]).size
    return args.width, args.height


def cmd_infer(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = weight_dtype(args.mixed_precision) if device.type == "cuda" else torch.float32
    model = load_production_model(args.checkpoint, map_location=device).to(
        device=device,
        dtype=dtype,
    )
    model.eval()

    target_size = _infer_target_size(args)
    source, source_flag = _load_condition_sequence(
        _paths_from_input(args.source),
        num_frames=args.num_frames,
        target_size=target_size,
    )
    first, first_flag = _load_condition_sequence(
        _paths_from_input(args.first_frame),
        num_frames=args.num_frames,
        target_size=target_size,
    )
    reference_arg = args.reference or args.aux
    reference, ref_flag = _load_condition_sequence(
        _paths_from_input(reference_arg),
        num_frames=args.num_frames,
        target_size=target_size,
    )
    (
        source,
        first,
        reference,
        source_flag,
        first_flag,
        ref_flag,
    ) = _resize_channels(
        [source, first, reference, source_flag, first_flag, ref_flag],
        height=args.height,
        width=args.width,
        random_crop=False,
    )
    condition = torch.cat(
        [
            _normalize_image(source),
            _normalize_image(first),
            _normalize_image(reference),
            _normalize_mask(source_flag),
            _normalize_mask(first_flag),
            _normalize_mask(ref_flag),
        ],
        dim=1,
    ).permute(1, 0, 2, 3).unsqueeze(0)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and dtype != torch.float32,
    ):
        prediction = model(condition.to(device=device, dtype=dtype), [args.task], [args.prompt])[0]
    save_sequence_frames(prediction.float(), args.output)
    print(json.dumps({"output": args.output, "task": args.task}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anita production-stage shot model utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--tasks", default=",".join(TASKS))
    manifest.add_argument("--max-frames-per-shot", type=int, default=49)
    manifest.add_argument("--min-frames-per-shot", type=int, default=8)
    manifest.add_argument(
        "--shot-split-strategy",
        choices=SHOT_SPLIT_STRATEGIES,
        default="semantic",
    )
    manifest.add_argument("--semantic-split-search-radius", type=int, default=3)
    manifest.set_defaults(func=cmd_manifest)

    train_parser = subparsers.add_parser("train")
    for action in build_train_parser({})._actions:
        if action.dest == "help":
            continue
        flags = action.option_strings
        if not flags:
            continue
        kwargs = {
            "default": action.default,
            "type": action.type,
            "choices": action.choices,
            "required": action.required,
            "help": action.help,
            "nargs": action.nargs,
            "const": action.const,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if isinstance(action, argparse._StoreTrueAction):
            train_parser.add_argument(
                *flags,
                action="store_true",
                default=action.default,
                help=action.help,
            )
        else:
            train_parser.add_argument(*flags, **kwargs)
    train_parser.set_defaults(func=cmd_train)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--task", choices=TASKS, required=True)
    infer.add_argument("--prompt", default="")
    infer.add_argument("--source", help="Required shot for stages 2/3; optional for stage 1.")
    infer.add_argument("--first-frame", help="Optional first-frame conditioning image.")
    infer.add_argument(
        "--reference",
        help="Optional character/composition reference image or directory.",
    )
    infer.add_argument("--aux", help="Deprecated alias for --reference.")
    infer.add_argument(
        "--output",
        required=True,
        help="Output frame directory or contact-sheet image path.",
    )
    infer.add_argument("--num-frames", type=int, default=12)
    infer.add_argument("--height", type=int, default=384)
    infer.add_argument("--width", type=int, default=384)
    infer.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    infer.set_defaults(func=cmd_infer)
    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        args = parse_train_args(sys.argv[2:])
        raise SystemExit(cmd_train(args))
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
