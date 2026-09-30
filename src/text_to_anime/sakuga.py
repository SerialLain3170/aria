from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .captioning import clean_text
from .manifest import write_jsonl

CAPTION_KEYS = (
    "caption",
    "text",
    "video_text",
    "video_caption",
    "description",
    "short_caption",
    "dense_caption",
    "blip_caption",
    "gpt_caption",
    "llm_caption",
)

PATH_KEYS = ("video_path", "clip_path", "file_path", "path", "relative_path", "mp4_path")
SOURCE_KEYS = ("source_id", "video_id", "source_video", "youtube_id", "url", "hash_id", "clip_id")

NUMERIC_KEYS = (
    "width",
    "height",
    "fps",
    "duration",
    "seconds",
    "aesthetic_score",
    "cafe_aesthetic",
    "motion_score",
    "dynamicity",
    "dynamicity_ratio",
    "text_probability",
    "text_prob",
)

TAG_KEYS = ("tags", "anime_tags", "tag_string", "wd14_tags")


def iter_parquet_rows(parquet_paths: list[Path]) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Reading Sakuga parquet files requires pyarrow. Install with `pip install pyarrow`.") from exc

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            yield row


def iter_jsonl_rows(jsonl_paths: list[Path]) -> Iterable[dict[str, Any]]:
    for jsonl_path in jsonl_paths:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def find_caption(row: dict[str, Any]) -> str:
    for key in CAPTION_KEYS:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def find_path(row: dict[str, Any], clips_root: Path | None) -> str:
    for key in PATH_KEYS:
        value = clean_text(row.get(key))
        if not value:
            continue
        path = Path(value)
        if path.is_absolute() or clips_root is None:
            return str(path)
        return str(clips_root / path)
    clip_id = clean_text(row.get("clip_id") or row.get("id") or row.get("hash_id"))
    if clip_id and clips_root is not None:
        suffix = "" if clip_id.endswith((".mp4", ".webm", ".mkv")) else ".mp4"
        return str(clips_root / f"{clip_id}{suffix}")
    return ""


def find_source_id(row: dict[str, Any], fallback: str) -> str:
    for key in SOURCE_KEYS:
        value = clean_text(row.get(key))
        if value:
            return value
    return fallback


def normalize_tags(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(clean_text(item) for item in value if clean_text(item))
    if isinstance(value, dict):
        return ", ".join(key for key, enabled in value.items() if enabled)
    return clean_text(value)


def sakuga_row_to_manifest(row: dict[str, Any], *, clips_root: Path | None, index: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "video_path": find_path(row, clips_root),
        "caption": find_caption(row),
        "source_id": find_source_id(row, fallback=f"sakuga-source-{index:08d}"),
        "clip_id": clean_text(row.get("clip_id") or row.get("id") or row.get("hash_id") or f"sakuga-{index:08d}"),
    }

    for key in NUMERIC_KEYS:
        if key in row and row[key] not in (None, ""):
            record[key] = row[key]

    for key in TAG_KEYS:
        tags = normalize_tags(row.get(key))
        if tags:
            record["anime_tags"] = tags
            break

    for key in ("category", "content_type", "composition", "camera", "action", "style", "media", "venue"):
        value = clean_text(row.get(key))
        if value:
            record[key] = value

    if "motion_amplitude" not in record:
        for key in ("motion_score", "dynamicity", "dynamicity_ratio"):
            if key in record:
                record["motion_amplitude"] = record[key]
                break

    return record


def discover_inputs(input_path: str, *, kind: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    suffixes = {"parquet": "*.parquet", "jsonl": "*.jsonl"}
    return sorted(path.glob(suffixes[kind]))


def passes_basic_filters(record: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.require_caption and not clean_text(record.get("caption")):
        return False
    if args.require_video_path and not clean_text(record.get("video_path")):
        return False

    def number(name: str, default: float) -> float:
        try:
            return float(record.get(name, default))
        except (TypeError, ValueError):
            return default

    width = number("width", args.min_width)
    height = number("height", args.min_height)
    duration = number("duration", number("seconds", args.min_duration))
    if width < args.min_width or height < args.min_height:
        return False
    if duration < args.min_duration or duration > args.max_duration:
        return False
    return True


def cmd_build(args: argparse.Namespace) -> int:
    clips_root = Path(args.clips_root) if args.clips_root else None
    if args.input_format == "parquet":
        paths = discover_inputs(args.input, kind="parquet")
        rows = iter_parquet_rows(paths)
    else:
        paths = discover_inputs(args.input, kind="jsonl")
        rows = iter_jsonl_rows(paths)
    if not paths:
        raise FileNotFoundError(f"no {args.input_format} files found at {args.input}")

    records = []
    for index, row in enumerate(rows):
        record = sakuga_row_to_manifest(row, clips_root=clips_root, index=index)
        if passes_basic_filters(record, args):
            records.append(record)
        if args.limit and len(records) >= args.limit:
            break

    write_jsonl(args.output, records)
    print(json.dumps({"input_files": len(paths), "records": len(records), "output": args.output}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Sakuga-42M metadata into the text-to-anime JSONL manifest.")
    parser.add_argument("--input", required=True, help="Parquet file/dir from Sakuga Aesthetic, or JSONL file/dir.")
    parser.add_argument("--input-format", choices=("parquet", "jsonl"), default="parquet")
    parser.add_argument("--clips-root", help="Root directory containing downloaded/split Sakuga clips.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-width", type=int, default=360)
    parser.add_argument("--min-height", type=int, default=240)
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--require-caption", action="store_true", default=True)
    parser.add_argument("--allow-missing-caption", action="store_false", dest="require_caption")
    parser.add_argument("--require-video-path", action="store_true", default=True)
    parser.add_argument("--allow-missing-video-path", action="store_false", dest="require_video_path")
    parser.set_defaults(func=cmd_build)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
