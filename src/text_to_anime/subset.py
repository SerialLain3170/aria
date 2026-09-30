from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .captioning import build_caption, clean_text
from .manifest import read_jsonl, source_key, stable_bucket, write_jsonl

VN_TARGETS = {
    "talking_facial_acting": 0.30,
    "subtle_idle_motion": 0.20,
    "walking_turning_entering": 0.15,
    "emotional_gestures": 0.15,
    "camera_movement_composition": 0.10,
    "dynamic_action": 0.10,
}

CATEGORY_KEYWORDS = {
    "talking_facial_acting": (
        "talk", "speak", "speaking", "mouth", "dialogue", "conversation", "smile", "frown",
        "expression", "blush", "cry", "laugh", "whisper", "close-up", "close up",
    ),
    "subtle_idle_motion": (
        "blink", "breath", "breathing", "idle", "standing", "sitting", "stares", "looking",
        "hair moves", "wind", "gently", "slightly", "subtle",
    ),
    "walking_turning_entering": (
        "walk", "walking", "turn", "turns", "enter", "enters", "steps", "approaches",
        "leaves", "runs into", "moves toward",
    ),
    "emotional_gestures": (
        "raises her hand", "raises his hand", "gesture", "points", "nod", "nods", "shakes",
        "hands", "clenches", "reaches", "covers", "wipes", "bows",
    ),
    "camera_movement_composition": (
        "pan", "zoom", "push-in", "push in", "pull back", "tilt", "tracking", "camera",
        "close-up", "wide shot", "medium shot",
    ),
    "dynamic_action": (
        "fight", "battle", "attack", "explosion", "jump", "leap", "slash", "punch", "kick",
        "magic", "beam", "fall", "chase", "running",
    ),
}

BAD_TEXT_KEYWORDS = ("subtitle", "subtitles", "caption", "credits", "watermark", "logo", "split screen")


def get_number(record: dict[str, Any], names: tuple[str, ...], default: float | None = None) -> float | None:
    for name in names:
        value = record.get(name)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def infer_category(record: dict[str, Any]) -> str:
    explicit = clean_text(record.get("content_type") or record.get("category") or record.get("motion_category"))
    if explicit in VN_TARGETS:
        return explicit

    text = " ".join(
        clean_text(record.get(key)).lower()
        for key in ("caption", "dense_caption", "short_caption", "tags", "anime_tags", "camera", "action")
    )
    scores = {
        category: sum(1 for keyword in keywords if keyword in text)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    best_category, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_category
    motion = get_number(record, ("motion_amplitude", "motion_score", "dynamicity", "dynamicity_ratio"), 2.0)
    if motion is not None and motion >= 4.0:
        return "dynamic_action"
    return "subtle_idle_motion"


def quality_score(record: dict[str, Any]) -> float:
    caption = build_caption(record, include_scores=False).lower()
    score = 0.0

    width = get_number(record, ("width", "video_width", "w"), 0.0) or 0.0
    height = get_number(record, ("height", "video_height", "h"), 0.0) or 0.0
    duration = get_number(record, ("duration", "seconds", "clip_duration"), 4.0) or 4.0
    aesthetic = get_number(record, ("aesthetic_score", "aesthetic", "cafe_aesthetic"), 5.0) or 5.0
    motion = get_number(record, ("motion_amplitude", "motion_score", "dynamicity", "dynamicity_ratio"), 2.0) or 2.0
    text_prob = get_number(record, ("text_probability", "text_prob", "ocr_prob"), 0.0) or 0.0

    if width >= 640 and height >= 360:
        score += 2.0
    elif width >= 480 and height >= 270:
        score += 1.0
    else:
        score -= 2.0

    if 2.0 <= duration <= 6.5:
        score += 2.0
    elif 1.5 <= duration < 2.0 or 6.5 < duration <= 8.0:
        score += 0.5
    else:
        score -= 2.0

    score += min(max(aesthetic - 4.5, -1.0), 2.5)
    if 1.0 <= motion <= 3.5:
        score += 1.5
    elif motion < 1.0:
        score += 0.5
    else:
        score -= 0.75

    if any(keyword in caption for keyword in BAD_TEXT_KEYWORDS) or text_prob > 0.35:
        score -= 4.0
    if len(caption.split()) >= 18:
        score += 0.75
    if "camera" in caption or any(word in caption for word in ("push", "pan", "zoom", "close-up", "medium shot")):
        score += 0.5
    return score


def source_limited(records: list[dict[str, Any]], *, max_per_source: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    kept = []
    for record in records:
        key = source_key(record)
        if counts[key] >= max_per_source:
            continue
        counts[key] += 1
        kept.append(record)
    return kept


def select_subset(
    records: list[dict[str, Any]],
    *,
    target_size: int,
    min_quality: float,
    max_per_source: int,
) -> list[dict[str, Any]]:
    prepared = []
    for record in records:
        normalized = dict(record)
        normalized["content_type"] = infer_category(normalized)
        normalized["caption"] = build_caption(normalized)
        normalized["selection_score"] = quality_score(normalized)
        if normalized["selection_score"] >= min_quality:
            prepared.append(normalized)

    prepared.sort(key=lambda item: (-item["selection_score"], stable_bucket(str(item.get("video_path", item)))))

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for category, ratio in VN_TARGETS.items():
        quota = int(math.floor(target_size * ratio))
        candidates = [record for record in prepared if record["content_type"] == category]
        candidates = source_limited(candidates, max_per_source=max_per_source)
        for record in candidates[:quota]:
            selected.append(record)
            selected_ids.add(id(record))

    if len(selected) < target_size:
        remainder = [record for record in prepared if id(record) not in selected_ids]
        remainder = source_limited(remainder, max_per_source=max_per_source)
        selected.extend(remainder[: target_size - len(selected)])

    return selected[:target_size]


def cmd_select(args: argparse.Namespace) -> int:
    if args.target_size < 20000 or args.target_size > 50000:
        raise ValueError("target size should be in the requested 20K-50K range")
    records = read_jsonl(args.manifest)
    subset = select_subset(
        records,
        target_size=args.target_size,
        min_quality=args.min_quality,
        max_per_source=args.max_per_source,
    )
    write_jsonl(args.output, subset)
    counts = Counter(record["content_type"] for record in subset)
    print(json.dumps({"selected": len(subset), "by_content_type": dict(counts)}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a VN-oriented 20K-50K subset from Sakuga Aesthetic metadata.")
    parser.add_argument("select", nargs="?")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-size", type=int, default=30000)
    parser.add_argument("--min-quality", type=float, default=1.0)
    parser.add_argument("--max-per-source", type=int, default=12)
    parser.set_defaults(func=cmd_select)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
