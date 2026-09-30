from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .captioning import assess_caption, build_caption


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def source_key(record: dict[str, Any]) -> str:
    for key in ("source_id", "title", "episode", "source_video", "video_path"):
        value = record.get(key)
        if value:
            return str(value)
    raise ValueError(f"record is missing source key fields: {record}")


def stable_bucket(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def split_by_source(
    records: list[dict[str, Any]],
    *,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, val = [], []
    for record in records:
        target = val if stable_bucket(source_key(record)) < val_ratio else train
        target.append(record)
    return train, val


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized["caption"] = build_caption(normalized)
    return normalized


def validate_records(records: list[dict[str, Any]], *, require_files: bool) -> Counter:
    counts: Counter = Counter()
    for idx, record in enumerate(records):
        path = record.get("video_path")
        if not path:
            counts["missing_video_path"] += 1
        elif require_files and not Path(path).exists():
            counts["missing_file"] += 1

        quality = assess_caption(record)
        if not quality.caption:
            counts["missing_caption"] += 1
        if quality.missing:
            counts["incomplete_structured_caption"] += 1

        if "motion_amplitude" in record:
            try:
                score = float(record["motion_amplitude"])
            except (TypeError, ValueError):
                counts["bad_motion_amplitude"] += 1
            else:
                if not 0 <= score <= 5:
                    counts["bad_motion_amplitude"] += 1

        if idx % 10000 == 0:
            counts["checked"] = idx + 1
    counts["checked"] = len(records)
    return counts


def cmd_validate(args: argparse.Namespace) -> int:
    records = read_jsonl(args.manifest)
    counts = validate_records(records, require_files=args.require_files)
    print(json.dumps(dict(counts), indent=2, sort_keys=True))
    failures = sum(value for key, value in counts.items() if key not in {"checked", "incomplete_structured_caption"})
    return 1 if failures else 0


def cmd_normalize(args: argparse.Namespace) -> int:
    records = [normalize_record(record) for record in read_jsonl(args.manifest)]
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} records to {args.output}")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    records = [normalize_record(record) for record in read_jsonl(args.manifest)]
    train, val = split_by_source(records, val_ratio=args.val_ratio)
    write_jsonl(args.train_out, train)
    write_jsonl(args.val_out, val)
    print(json.dumps({"train": len(train), "val": len(val)}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manifest utilities for anime video training data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--require-files", action="store_true")
    validate.set_defaults(func=cmd_validate)

    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--manifest", required=True)
    normalize.add_argument("--output", required=True)
    normalize.set_defaults(func=cmd_normalize)

    split = subparsers.add_parser("split")
    split.add_argument("--manifest", required=True)
    split.add_argument("--train-out", required=True)
    split.add_argument("--val-out", required=True)
    split.add_argument("--val-ratio", type=float, default=0.02)
    split.set_defaults(func=cmd_split)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()

