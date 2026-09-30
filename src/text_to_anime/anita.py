from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable

from .manifest import write_jsonl

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SEQUENCE_TYPES = {"sketch", "color", "composition"}


def iter_sequence_dirs(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for directory in sorted(root.rglob("*")):
        if not directory.is_dir():
            continue
        has_images = any(
            path.suffix.lower() in IMAGE_EXTS for path in directory.iterdir() if path.is_file()
        )
        if not has_images:
            continue
        if (
            directory.name.lower() in SEQUENCE_TYPES
            or directory.parent.name.lower() in SEQUENCE_TYPES
        ):
            yield directory


def sequence_identity(sequence_dir: Path, root: Path) -> tuple[str, str, str]:
    relative = sequence_dir.relative_to(root)
    parts = relative.parts
    if sequence_dir.name.lower() in SEQUENCE_TYPES:
        seq_type = sequence_dir.name.lower()
        work = parts[0] if len(parts) >= 1 else "AnitaDataset"
        scene = "/".join(parts[1:-1]) if len(parts) > 2 else sequence_dir.parent.name
    else:
        work = parts[0] if len(parts) >= 1 else "AnitaDataset"
        seq_type = (
            sequence_dir.parent.name.lower()
            if sequence_dir.parent.name.lower() in SEQUENCE_TYPES
            else "sequence"
        )
        scene = sequence_dir.name if len(parts) >= 1 else "scene"
    return work, scene, seq_type


def sequence_caption(work: str, scene: str, seq_type: str) -> str:
    label = {
        "sketch": "tie-down sketch animation sequence",
        "color": "segmented color animation sequence",
        "composition": "composited finished animation sequence",
    }.get(seq_type, f"{seq_type} animation sequence")
    return (
        f"A 2D hand-drawn industrial cartoon {label} from {work}, {scene}, "
        "shown as a continuous shot. "
        "There is no text in the video."
    )


def output_path_for(output_root: Path | None, work: str, scene: str, seq_type: str) -> str:
    if output_root is None:
        return ""
    safe_work = work.replace("/", "_").replace(" ", "_")
    safe_scene = scene.replace("/", "_").replace(" ", "_")
    return str(output_root / safe_work / f"{safe_scene}_{seq_type}.mp4")


def build_records(
    root: str | Path,
    *,
    clips_root: str | Path | None = None,
    fps: int = 12,
) -> list[dict]:
    root = Path(root)
    clip_root_path = Path(clips_root) if clips_root else None
    records = []
    for index, sequence_dir in enumerate(iter_sequence_dirs(root)):
        work, scene, seq_type = sequence_identity(sequence_dir, root)
        frames = sorted(
            path for path in sequence_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
        )
        record = {
            "video_path": output_path_for(clip_root_path, work, scene, seq_type),
            "caption": sequence_caption(work, scene, seq_type),
            "source_id": f"AnitaDataset:{work}",
            "clip_id": f"anita_{index:06d}",
            "dataset": "AnitaDataset",
            "content_type": "subtle_idle_motion",
            "style": "2D hand-drawn industrial animation",
            "sequence_type": seq_type,
            "work": work,
            "scene": scene,
            "frame_dir": str(sequence_dir),
            "num_source_frames": len(frames),
            "fps": fps,
        }
        if frames:
            record["first_frame"] = str(frames[0])
            record["last_frame"] = str(frames[-1])
        records.append(record)
    return records


def render_sequence(frame_dir: Path, output: Path, fps: int, *, overwrite: bool) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not frames:
        return False
    concat_file = output.with_suffix(".frames.txt")
    with concat_file.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(f"file '{frame.resolve()}'\n")
            handle.write(f"duration {1.0 / fps:.6f}\n")
        handle.write(f"file '{frames[-1].resolve()}'\n")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-r", str(fps), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
    ]
    result = subprocess.run(cmd, check=False)
    concat_file.unlink(missing_ok=True)
    return result.returncode == 0 and output.exists()


def cmd_manifest(args: argparse.Namespace) -> int:
    records = build_records(args.root, clips_root=args.clips_root, fps=args.fps)
    write_jsonl(args.output, records)
    print(json.dumps({"records": len(records), "output": args.output}, indent=2, sort_keys=True))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    records = build_records(args.root, clips_root=args.clips_root, fps=args.fps)
    rendered = []
    failed = 0
    for index, record in enumerate(records):
        if args.limit and index >= args.limit:
            break
        if render_sequence(
            Path(record["frame_dir"]),
            Path(record["video_path"]),
            args.fps,
            overwrite=args.overwrite,
        ):
            rendered.append(record)
        else:
            failed += 1
    if args.output_manifest:
        write_jsonl(args.output_manifest, rendered)
    print(json.dumps({"rendered": len(rendered), "failed": failed}, indent=2, sort_keys=True))
    return 1 if failed and args.strict else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build manifests and clips from AnitaDataset image sequences."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--clips-root")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--fps", type=int, default=12)
    manifest.set_defaults(func=cmd_manifest)

    render = subparsers.add_parser("render-clips")
    render.add_argument("--root", required=True)
    render.add_argument("--clips-root", required=True)
    render.add_argument("--output-manifest")
    render.add_argument("--fps", type=int, default=12)
    render.add_argument("--limit", type=int)
    render.add_argument("--overwrite", action="store_true")
    render.add_argument("--strict", action="store_true")
    render.set_defaults(func=cmd_render)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
