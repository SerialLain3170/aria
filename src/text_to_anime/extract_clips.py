from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .manifest import read_jsonl, write_jsonl


def ffmpeg_extract(source: str, output: str, start: float, end: float | None, *, overwrite: bool) -> bool:
    source_path = Path(source)
    output_path = Path(output)
    if not source_path.exists():
        return False
    if output_path.exists() and not overwrite:
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source_path),
    ]
    if end is not None and end > start:
        cmd.extend(["-to", f"{end - start:.3f}"])
    cmd.extend(["-map", "0:v:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", str(output_path)])
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0 and output_path.exists()


def cmd_extract(args: argparse.Namespace) -> int:
    records = read_jsonl(args.manifest)
    ok_records = []
    missing = 0
    failed = 0
    for index, record in enumerate(records):
        if args.limit and index >= args.limit:
            break
        source = record.get("source_video_path")
        output = record.get("video_path")
        if not source or not output:
            missing += 1
            continue
        try:
            start = float(record.get("start_time", 0.0))
        except (TypeError, ValueError):
            start = 0.0
        end = record.get("end_time")
        try:
            end_value = float(end) if end is not None else None
        except (TypeError, ValueError):
            end_value = None
        if ffmpeg_extract(source, output, start, end_value, overwrite=args.overwrite):
            ok_records.append(record)
        else:
            failed += 1
    if args.output_manifest:
        write_jsonl(args.output_manifest, ok_records)
    print({"clips_ok": len(ok_records), "missing_paths": missing, "failed": failed})
    return 1 if failed and args.strict else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract timestamped source videos into training clips using ffmpeg.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", help="Optional manifest containing only successfully extracted clips.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.set_defaults(func=cmd_extract)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
