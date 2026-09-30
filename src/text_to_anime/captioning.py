from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CAPTION_FIELDS = (
    "subject",
    "appearance",
    "initial_state",
    "action",
    "secondary_motion",
    "camera",
    "background",
    "style",
)


@dataclass(frozen=True)
class CaptionQuality:
    ok: bool
    missing: tuple[str, ...]
    caption: str


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def build_caption(record: dict[str, Any], *, include_scores: bool = True) -> str:
    caption = clean_text(record.get("caption"))
    if caption:
        return append_generation_clauses(caption, record, include_scores=include_scores)

    parts = []
    subject = clean_text(record.get("subject"))
    appearance = clean_text(record.get("appearance"))
    initial_state = clean_text(record.get("initial_state"))
    action = clean_text(record.get("action"))
    secondary_motion = clean_text(record.get("secondary_motion"))
    camera = clean_text(record.get("camera"))
    background = clean_text(record.get("background"))
    style = clean_text(record.get("style"))

    if background:
        parts.append(f"In {background}")
    if subject and appearance:
        parts.append(f"{subject} with {appearance}")
    elif subject:
        parts.append(subject)
    elif appearance:
        parts.append(f"A character with {appearance}")
    if initial_state:
        parts.append(initial_state)
    if action:
        parts.append(action)
    if secondary_motion:
        parts.append(f"while {secondary_motion}")
    if camera:
        parts.append(camera)
    if style:
        parts.append(style)

    generated = ", ".join(parts).strip(" ,")
    if generated and generated[-1] not in ".!?":
        generated += "."
    return append_generation_clauses(generated, record, include_scores=include_scores)


def append_generation_clauses(caption: str, record: dict[str, Any], *, include_scores: bool) -> str:
    caption = clean_text(caption)
    additions = []

    if include_scores:
        aesthetic_score = record.get("aesthetic_score")
        motion_score = record.get("motion_score", record.get("motion_amplitude"))
        if aesthetic_score is not None:
            additions.append(f"aesthetic score: {float(aesthetic_score):.1f}.")
        if motion_score is not None:
            additions.append(f"motion score: {float(motion_score):.1f}.")

    no_text = clean_text(record.get("no_text_clause", "There is no text in the video."))
    if no_text and no_text.lower() not in caption.lower():
        additions.append(no_text if no_text[-1] in ".!?" else f"{no_text}.")

    if additions:
        caption = f"{caption.rstrip()} {' '.join(additions)}"
    return caption.strip()


def assess_caption(record: dict[str, Any]) -> CaptionQuality:
    caption = build_caption(record, include_scores=False)
    missing = tuple(field for field in CAPTION_FIELDS if not clean_text(record.get(field)))
    return CaptionQuality(ok=bool(caption) and len(missing) <= 3, missing=missing, caption=caption)

