from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .manifest import read_jsonl, write_jsonl
from .video import load_video_frames


def tensor_frame_to_image(frame: torch.Tensor) -> Image.Image:
    if frame.ndim != 3:
        raise ValueError(f"expected CHW frame, got {tuple(frame.shape)}")
    frame = frame.detach().cpu().clamp(0, 255).to(torch.uint8)
    return Image.fromarray(frame.permute(1, 2, 0).numpy())


def frame_output_path(record: dict, output_root: Path, *, field: str, suffix: str) -> Path:
    clip_id = record.get("clip_id") or Path(record.get("video_path", "clip")).stem
    return output_root / f"{clip_id}_{field}{suffix}"


def extract_frame_image(video_path: str | Path, *, frame_index: int) -> Image.Image:
    frames = load_video_frames(video_path)
    if frame_index < 0:
        frame_index = frames.shape[0] + frame_index
    frame_index = max(0, min(frame_index, frames.shape[0] - 1))
    return tensor_frame_to_image(frames[frame_index])


def cmd_extract(args: argparse.Namespace) -> int:
    records = read_jsonl(args.manifest)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    updated = []
    failures = 0
    field = "first_frame_path" if args.frame_index == 0 else args.field
    for index, record in enumerate(records):
        if args.limit and index >= args.limit:
            break
        video_path = record.get("video_path")
        if not video_path or not Path(video_path).exists():
            failures += 1
            if args.keep_failed:
                updated.append(record)
            continue
        try:
            image = extract_frame_image(video_path, frame_index=args.frame_index)
            output_path = frame_output_path(record, output_root, field=field, suffix=args.suffix)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=args.quality)
            next_record = dict(record)
            next_record[field] = str(output_path)
            updated.append(next_record)
        except Exception:
            failures += 1
            if args.keep_failed:
                updated.append(record)
    write_jsonl(args.output_manifest, updated)
    print(json.dumps({"records": len(updated), "failures": failures, "output": args.output_manifest}, indent=2, sort_keys=True))
    return 1 if failures and args.strict else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract conditioning frames from video manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frame-index", type=int, default=0, help="0 for first frame, -1 for last frame.")
    parser.add_argument("--field", default="conditioning_frame_path")
    parser.add_argument("--suffix", default=".jpg")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--keep-failed", action="store_true")
    parser.set_defaults(func=cmd_extract)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
