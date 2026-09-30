from __future__ import annotations

import argparse
from pathlib import Path

from .manifest import read_jsonl


def youtube_url(value: str) -> str:
    value = value.strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"https://www.youtube.com/watch?v={value}"


def cmd_export(args: argparse.Namespace) -> int:
    records = read_jsonl(args.manifest)
    ids = []
    seen = set()
    for record in records:
        value = str(record.get(args.field) or record.get("source_url") or "").strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        ids.append(youtube_url(value) if args.as_urls else value)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    print({"ids": len(ids), "output": str(output)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export unique source video IDs/URLs from a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--field", default="source_id")
    parser.add_argument("--as-urls", action="store_true")
    parser.set_defaults(func=cmd_export)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
