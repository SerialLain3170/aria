from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from tqdm.auto import tqdm

from .anita_production import iter_shot_sets
from .manifest import write_jsonl

# Anita ships each shot as a character-color layer (RGBA, transparent background) and the final
# composition (opaque). Wherever the character alpha is empty, the composition pixel is background,
# so a background plate can be recovered per shot:
#   - static camera: per-pixel median of the visible background over time -> one clean plate
#   - moving camera / animated background: per-frame background with character holes filled
# The per-frame backgrounds are always written, so downstream code can use them for either case.


def load_rgba(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def working_size(path: str | Path, max_side: int) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def character_mask(alpha: np.ndarray, *, threshold: int, dilate: int) -> np.ndarray:
    mask = alpha > threshold
    if dilate > 0 and mask.any():
        # Dilate to drop anti-aliased edges and outline pixels that sit just outside the alpha.
        image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        image = image.filter(ImageFilter.MaxFilter(2 * dilate + 1))
        mask = np.asarray(image) > 127
    return mask


def resize_float_rgb(rgb: np.ndarray, *, width: int, height: int) -> np.ndarray:
    channels = [
        np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., c]), mode="F").resize((width, height), Image.Resampling.BILINEAR))
        for c in range(rgb.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def push_pull_fill(rgb: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Fill unknown pixels with a smooth pyramid interpolation of the known ones."""
    if known.all():
        return rgb.astype(np.float32)
    if not known.any():
        return np.full(rgb.shape, 127.5, dtype=np.float32)
    levels: list[tuple[np.ndarray, np.ndarray]] = []
    color = rgb.astype(np.float32) * known[..., None]
    weight = known.astype(np.float32)
    while True:
        levels.append((color, weight))
        height, width = weight.shape
        if (weight > 0).all() or max(height, width) <= 1:
            break
        pad_h, pad_w = height % 2, width % 2
        color = np.pad(color, ((0, pad_h), (0, pad_w), (0, 0)))
        weight = np.pad(weight, ((0, pad_h), (0, pad_w)))
        color = color.reshape(color.shape[0] // 2, 2, color.shape[1] // 2, 2, 3).sum(axis=(1, 3))
        weight = weight.reshape(weight.shape[0] // 2, 2, weight.shape[1] // 2, 2).sum(axis=(1, 3))
    color, weight = levels[-1]
    filled = color / np.maximum(weight[..., None], 1e-6)
    for color, weight in reversed(levels[:-1]):
        height, width = weight.shape
        upsampled = resize_float_rgb(filled, width=width, height=height)
        known_color = color / np.maximum(weight[..., None], 1e-6)
        filled = np.where(weight[..., None] > 0, known_color, upsampled)
    return filled


def masked_median(frames: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stack = frames.astype(np.float32)
    stack[~known] = np.nan
    coverage = known.any(axis=0)
    median = np.zeros(frames.shape[1:], dtype=np.float32)
    with np.errstate(all="ignore"):
        values = np.nanmedian(stack[:, coverage], axis=0)
    median[coverage] = values
    return median, coverage


def build_scene_plate(
    *,
    color_paths: dict[str, str],
    composition_paths: dict[str, str],
    output_dir: Path,
    max_side: int,
    alpha_threshold: int,
    dilate: int,
    max_median_frames: int,
    static_threshold: float,
) -> dict[str, Any]:
    frame_ids = sorted(composition_paths)
    size = working_size(composition_paths[frame_ids[0]], max_side)
    width, height = size
    frames = []
    masks = []
    for frame_id in frame_ids:
        composition = load_rgba(composition_paths[frame_id], size)[..., :3]
        if frame_id in color_paths:
            alpha = load_rgba(color_paths[frame_id], size)[..., 3]
        else:
            # Composition drawings with no color layer have no character in them.
            alpha = np.zeros((height, width), dtype=np.uint8)
        frames.append(composition)
        masks.append(character_mask(alpha, threshold=alpha_threshold, dilate=dilate))
    frame_stack = np.stack(frames)
    known = ~np.stack(masks)

    sample = np.unique(np.linspace(0, len(frame_ids) - 1, min(max_median_frames, len(frame_ids))).round().astype(int))
    plate, coverage = masked_median(frame_stack[sample], known[sample])
    plate = push_pull_fill(plate, coverage)

    # Mean absolute deviation of the visible background from the plate, in 0-255 units.
    residuals = []
    for frame, frame_known in zip(frame_stack, known):
        if frame_known.any():
            residuals.append(float(np.abs(frame[frame_known].astype(np.float32) - plate[frame_known]).mean()))
    residual = float(np.median(residuals)) if residuals else 0.0
    static = residual <= static_threshold

    output_dir.mkdir(parents=True, exist_ok=True)
    plate_path = output_dir / "plate.png"
    Image.fromarray(plate.round().clip(0, 255).astype(np.uint8)).save(plate_path)
    Image.fromarray(coverage.astype(np.uint8) * 255, mode="L").save(output_dir / "plate_coverage.png")

    background_dir = output_dir / "background"
    background_dir.mkdir(exist_ok=True)
    background_paths = {}
    for frame_id, frame, frame_known in zip(frame_ids, frame_stack, known):
        # Static shots borrow hidden pixels from the plate; moving shots interpolate within the frame.
        fill = plate if static else push_pull_fill(frame, frame_known)
        background = np.where(frame_known[..., None], frame.astype(np.float32), fill)
        path = background_dir / f"{frame_id}.png"
        Image.fromarray(background.round().clip(0, 255).astype(np.uint8)).save(path)
        background_paths[frame_id] = str(path)

    return {
        "plate_path": str(plate_path),
        "background_paths": background_paths,
        "static": static,
        "residual": residual,
        "plate_coverage": float(coverage.mean()),
        "character_coverage": float((~known).mean()),
        "width": width,
        "height": height,
        "num_frames": len(frame_ids),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover per-shot background plates from Anita color layers and compositions."
    )
    parser.add_argument("--root", default="/data/shasegawa/t2a/datasets/anita")
    parser.add_argument("--output-root", default="/data/shasegawa/t2a/datasets/anita/background_plates")
    parser.add_argument("--index", default="/data/shasegawa/t2a/manifests/anita_background_plates.jsonl")
    parser.add_argument("--scenes", help="Optional comma-separated scene names (e.g. 119_a,213_a).")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--alpha-threshold", type=int, default=8)
    parser.add_argument("--dilate", type=int, default=3, help="Character mask dilation in working pixels.")
    parser.add_argument("--max-median-frames", type=int, default=48)
    parser.add_argument(
        "--static-threshold",
        type=float,
        default=6.0,
        help="Max median background deviation from the plate (0-255) to treat the camera as static.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    wanted = {item.strip() for item in args.scenes.split(",")} if args.scenes else None
    shots = [
        shot
        for shot in iter_shot_sets(args.root)
        if shot.composition_paths and shot.color_paths and (wanted is None or shot.scene in wanted)
    ]
    rows = []
    for shot in tqdm(shots, desc="background plates"):
        info = build_scene_plate(
            color_paths=shot.color_paths,
            composition_paths=shot.composition_paths,
            output_dir=Path(args.output_root) / shot.work / shot.scene,
            max_side=args.max_side,
            alpha_threshold=args.alpha_threshold,
            dilate=args.dilate,
            max_median_frames=args.max_median_frames,
            static_threshold=args.static_threshold,
        )
        rows.append({"work": shot.work, "scene": shot.scene, **info})
        print(
            f"{shot.work}/{shot.scene}: static={info['static']} residual={info['residual']:.2f} "
            f"plate_coverage={info['plate_coverage']:.2f}"
        )
    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.index, rows)
    print(f"wrote {len(rows)} plates to {args.index}")


if __name__ == "__main__":
    main()
