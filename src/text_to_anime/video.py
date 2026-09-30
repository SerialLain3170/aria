from __future__ import annotations

import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_with_decord(path: str | Path) -> torch.Tensor | None:
    try:
        from decord import VideoReader, cpu
    except Exception:
        return None

    reader = VideoReader(str(path), ctx=cpu(0))
    if len(reader) == 0:
        raise ValueError(f"video has no frames: {path}")
    frames = reader.get_batch(list(range(len(reader)))).asnumpy()
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    return tensor


def _load_with_torchvision(path: str | Path) -> torch.Tensor:
    from torchvision.io import read_video

    frames, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
    if frames.numel() == 0:
        raise ValueError(f"video has no frames: {path}")
    return frames.contiguous()


def load_video_frames(path: str | Path) -> torch.Tensor:
    frames = _load_with_decord(path)
    if frames is not None:
        return frames
    return _load_with_torchvision(path)


def sample_indices(total_frames: int, num_frames: int, *, random_clip: bool) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if total_frames >= num_frames:
        if random_clip:
            start = random.randint(0, total_frames - num_frames)
        else:
            start = max((total_frames - num_frames) // 2, 0)
        return list(range(start, start + num_frames))

    if total_frames == 1:
        return [0] * num_frames

    positions = torch.linspace(0, total_frames - 1, steps=num_frames)
    return [int(round(position.item())) for position in positions]


def resize_and_crop(
    frames: torch.Tensor,
    *,
    height: int,
    width: int,
    random_crop: bool,
) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"expected TCHW frames, got shape {tuple(frames.shape)}")
    _, _, in_height, in_width = frames.shape
    scale = max(height / in_height, width / in_width)
    resized_height = max(height, math.ceil(in_height * scale))
    resized_width = max(width, math.ceil(in_width * scale))

    frames = F.interpolate(
        frames.float(),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )

    if random_crop:
        top = random.randint(0, resized_height - height) if resized_height > height else 0
        left = random.randint(0, resized_width - width) if resized_width > width else 0
    else:
        top = max((resized_height - height) // 2, 0)
        left = max((resized_width - width) // 2, 0)
    return frames[:, :, top : top + height, left : left + width]


def load_video_tensor(
    path: str | Path,
    *,
    num_frames: int,
    height: int,
    width: int,
    random_clip: bool = True,
    random_crop: bool = True,
) -> torch.Tensor:
    frames = load_video_frames(path)
    indices = sample_indices(frames.shape[0], num_frames, random_clip=random_clip)
    frames = frames[indices]
    frames = resize_and_crop(frames, height=height, width=width, random_crop=random_crop)
    frames = frames / 127.5 - 1.0
    return frames.permute(1, 0, 2, 3).contiguous()


def assert_wan_frame_count(num_frames: int) -> None:
    if (num_frames - 1) % 4 != 0:
        raise ValueError("Wan frame counts should satisfy num_frames = 4 * k + 1")

