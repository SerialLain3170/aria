from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .captioning import clean_text
from .manifest import write_jsonl

TIME_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def first_value(mapping: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return default


def parse_time(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = TIME_RE.match(text)
    if not match:
        return None
    hours = float(match.group(1) or 0)
    minutes = float(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def iter_annotation_files(path: str | Path) -> Iterable[tuple[str, dict[str, Any]]]:
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith(".json"):
                    with archive.open(name) as handle:
                        yield name, json.loads(handle.read().decode("utf-8"))
        return
    if path.is_file():
        yield str(path), json.loads(path.read_text(encoding="utf-8"))
        return
    for json_path in sorted(path.rglob("*.json")):
        yield str(json_path), json.loads(json_path.read_text(encoding="utf-8"))


def video_id_from_annotation(annotation: dict[str, Any], fallback_name: str) -> str:
    value = first_value(annotation, ("video ID", "video_id", "video id", "id", "youtube_id"))
    if value:
        return clean_text(value)
    return Path(fallback_name).stem


def annotation_fps(annotation: dict[str, Any]) -> float | None:
    value = first_value(annotation, ("fps", "frame_rate", "frame rate"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def get_story_script(segment: dict[str, Any]) -> dict[str, Any]:
    value = first_value(segment, ("story script", "story_script", "script"), {})
    return value if isinstance(value, dict) else {}


def get_visual_annotation(shot: dict[str, Any]) -> dict[str, Any]:
    value = first_value(shot, ("visual annotation", "visual_annotation", "visual"), {})
    return value if isinstance(value, dict) else {}


def scene_environment(story: dict[str, Any], scene_id: str) -> str:
    scenes = first_value(story, ("main scenes", "main_scenes", "scenes"), [])
    if not isinstance(scenes, list):
        return ""
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        sid = clean_text(first_value(scene, ("ID", "id", "scene_id")))
        if sid == scene_id:
            return clean_text(first_value(scene, ("environment", "description", "caption")))
    return ""


def character_appearances(story: dict[str, Any], character_ids: list[str]) -> str:
    characters = first_value(story, ("main characters", "main_characters", "characters"), [])
    if not isinstance(characters, list):
        return ""
    wanted = {clean_text(value) for value in character_ids}
    descriptions = []
    for character in characters:
        if not isinstance(character, dict):
            continue
        cid = clean_text(first_value(character, ("ID", "id", "character_id")))
        if wanted and cid not in wanted:
            continue
        appearance = clean_text(first_value(character, ("appearance", "description", "caption")))
        if appearance:
            descriptions.append(appearance)
    return "; ".join(descriptions)


def shot_caption(storyline: str, visual: dict[str, Any]) -> str:
    narrative = clean_text(first_value(visual, ("narrative caption", "narrative_caption", "narrative")))
    descriptive = clean_text(first_value(visual, ("descriptive caption", "descriptive_caption", "description")))
    if narrative and descriptive and descriptive.lower() not in narrative.lower():
        return f"{narrative} {descriptive}"
    if narrative:
        return narrative
    if descriptive:
        return descriptive
    return storyline


def clip_path_for(
    *,
    clips_root: Path | None,
    video_id: str,
    segment_index: int,
    shot_index: int,
    ext: str,
) -> str:
    if clips_root is None:
        return ""
    return str(clips_root / video_id / f"seg{segment_index:03d}_shot{shot_index:03d}{ext}")


def source_video_path_for(videos_root: Path | None, video_id: str) -> str:
    if videos_root is None:
        return ""
    for ext in (".mp4", ".webm", ".mkv", ".mov"):
        candidate = videos_root / f"{video_id}{ext}"
        if candidate.exists():
            return str(candidate)
    return str(videos_root / f"{video_id}.mp4")


def animeshooter_to_records(
    annotation: dict[str, Any],
    *,
    source_name: str,
    clips_root: Path | None,
    videos_root: Path | None,
    clip_ext: str,
) -> list[dict[str, Any]]:
    video_id = video_id_from_annotation(annotation, source_name)
    fps = annotation_fps(annotation)
    url = clean_text(first_value(annotation, ("url", "video_url", "source_url")))
    source_video_path = source_video_path_for(videos_root, video_id)
    segments = first_value(annotation, ("segments", "segment"), [])
    if not isinstance(segments, list):
        return []

    records: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        story = get_story_script(segment)
        storyline = clean_text(first_value(story, ("storyline", "story", "summary")))
        segment_start_frame = first_value(segment, ("start frame index", "start_frame_index", "start_frame"), 0) or 0
        try:
            segment_start_frame = int(segment_start_frame)
        except (TypeError, ValueError):
            segment_start_frame = 0
        segment_start_time = segment_start_frame / fps if fps else 0.0

        shots = first_value(story, ("shots", "shot"), [])
        if not isinstance(shots, list):
            continue
        for shot_index, shot in enumerate(shots):
            if not isinstance(shot, dict):
                continue
            visual = get_visual_annotation(shot)
            caption = shot_caption(storyline, visual)
            if not caption:
                continue
            start = parse_time(first_value(shot, ("start time", "start_time", "start")))
            end = parse_time(first_value(shot, ("end time", "end_time", "end")))
            absolute_start = segment_start_time + (start or 0.0)
            absolute_end = segment_start_time + end if end is not None else None
            duration = absolute_end - absolute_start if absolute_end is not None else None
            if duration is not None and duration <= 0:
                duration = None

            character_ids = first_value(shot, ("main characters", "main_characters", "characters"), [])
            if not isinstance(character_ids, list):
                character_ids = [character_ids]
            character_ids = [clean_text(item) for item in character_ids if clean_text(item)]
            scene_id = clean_text(first_value(shot, ("scene", "scene_id")))

            record: dict[str, Any] = {
                "video_path": clip_path_for(
                    clips_root=clips_root,
                    video_id=video_id,
                    segment_index=segment_index,
                    shot_index=shot_index,
                    ext=clip_ext,
                ),
                "caption": caption,
                "source_id": video_id,
                "clip_id": f"{video_id}_seg{segment_index:03d}_shot{shot_index:03d}",
                "dataset": "AnimeShooter",
                "content_type": "talking_facial_acting",
                "source_video_path": source_video_path,
                "source_url": url,
                "start_time": round(absolute_start, 3),
                "segment_index": segment_index,
                "shot_index": shot_index,
            }
            if absolute_end is not None:
                record["end_time"] = round(absolute_end, 3)
            if duration is not None:
                record["duration"] = round(duration, 3)
            if fps is not None:
                record["fps"] = fps
            appearance = character_appearances(story, character_ids)
            if appearance:
                record["appearance"] = appearance
            background = scene_environment(story, scene_id)
            if background:
                record["background"] = background
            if storyline:
                record["storyline"] = storyline
            narrative = clean_text(first_value(visual, ("narrative caption", "narrative_caption", "narrative")))
            descriptive = clean_text(first_value(visual, ("descriptive caption", "descriptive_caption", "description")))
            if narrative:
                record["narrative_caption"] = narrative
            if descriptive:
                record["descriptive_caption"] = descriptive
            records.append(record)
    return records


def cmd_build(args: argparse.Namespace) -> int:
    clips_root = Path(args.clips_root) if args.clips_root else None
    videos_root = Path(args.videos_root) if args.videos_root else None
    records: list[dict[str, Any]] = []
    files = 0
    for source_name, annotation in iter_annotation_files(args.annotations):
        files += 1
        records.extend(
            animeshooter_to_records(
                annotation,
                source_name=source_name,
                clips_root=clips_root,
                videos_root=videos_root,
                clip_ext=args.clip_ext,
            )
        )
        if args.limit and len(records) >= args.limit:
            records = records[: args.limit]
            break
    write_jsonl(args.output, records)
    print(json.dumps({"annotation_files": files, "records": len(records), "output": args.output}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert AnimeShooter annotations into shot-level JSONL manifests.")
    parser.add_argument("--annotations", required=True, help="dataset_anime_shooter.zip, a JSON file, or extracted annotation directory.")
    parser.add_argument("--clips-root", help="Root where extracted shot clips are or will be stored.")
    parser.add_argument("--videos-root", help="Root where source YouTube videos are or will be stored.")
    parser.add_argument("--clip-ext", default=".mp4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.set_defaults(func=cmd_build)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
