from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol

import torch
from PIL import Image
from tqdm.auto import tqdm

from .anita_production import build_production_samples
from .manifest import read_jsonl, write_jsonl
from .video import sample_indices

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_PROMPT = (
    "Describe this anime/cartoon shot for training a text-conditioned animation model. "
    "The images are representative frames from the same shot. Write one concise caption "
    "in English. Include the main subject, action or pose change, framing/camera if "
    "visible, background/setting, lighting or mood, and important visual attributes. "
    "Do not mention file names, frame numbers, that these are multiple images, or "
    "production-stage words such as sketch, color, or composition. If motion is unclear, "
    "describe the visible pose progression conservatively."
)
TASK_PREFIX = {
    "line_art": "Generate a line-art animation shot matching this content:",
    "character_color": "Colorize the main characters in this line-art animation shot:",
    "compose_refine": "Complete the final composited anime shot for this content:",
}


class Captioner(Protocol):
    model_path: str
    source: str

    def caption(self, image_paths: list[str], prompt: str) -> str:
        ...


def load_records(path_or_root: str | Path) -> list[dict[str, Any]]:
    path = Path(path_or_root)
    if path.is_file():
        return read_jsonl(path)
    return build_production_samples(path)


def record_shot_key(record: dict[str, Any]) -> str:
    work = str(record.get("work", "AnitaDataset"))
    scene = str(record.get("scene", "scene"))
    return f"{work}::{scene}"


def representative_paths(paths: list[str], *, num_frames: int) -> list[str]:
    if not paths:
        return []
    indices = sample_indices(len(paths), num_frames, random_clip=False)
    seen = set()
    selected = []
    for index in indices:
        path = paths[index]
        if path in seen:
            continue
        seen.add(path)
        selected.append(path)
    return selected


def best_visual_paths(records: Iterable[dict[str, Any]]) -> list[str]:
    records = list(records)
    for task in ("compose_refine", "character_color", "line_art"):
        for record in records:
            if record.get("task") == task and record.get("target_paths"):
                return list(record["target_paths"])
    for record in records:
        for key in ("reference_paths", "primary_paths", "target_paths"):
            if record.get(key):
                return list(record[key])
    return []


def group_by_shot(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record_shot_key(record)].append(record)
    return dict(grouped)


def clean_caption(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            for key in ("caption", "shot_caption", "description"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break
    text = " ".join(text.split())
    text = text.strip(' "')
    if text and text[-1] not in ".!?":
        text += "."
    return text


class LocalVLMCaptioner:
    source = "local_vlm"

    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        torch_dtype: str,
        trust_remote_code: bool,
        max_new_tokens: int,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        self.device = torch.device(resolved_device)
        dtype = self._dtype(torch_dtype)
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        self.model = ModelClass.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _dtype(name: str) -> torch.dtype | str:
        if name == "auto":
            return "auto"
        if name == "bf16":
            return torch.bfloat16
        if name == "fp16":
            return torch.float16
        if name == "fp32":
            return torch.float32
        raise ValueError(f"unknown torch dtype: {name}")

    def caption(self, image_paths: list[str], prompt: str) -> str:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        if not images:
            return ""
        inputs = self._prepare_inputs(images, prompt)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        input_ids = inputs.get("input_ids")
        if input_ids is not None and generated.shape[-1] > input_ids.shape[-1]:
            generated = generated[:, input_ids.shape[-1] :]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return clean_caption(text)

    def _prepare_inputs(self, images: list[Image.Image], prompt: str) -> dict[str, torch.Tensor]:
        if hasattr(self.processor, "apply_chat_template"):
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            try:
                inputs = self.processor(text=[text], images=images, return_tensors="pt")
            except TypeError:
                inputs = self.processor(text=text, images=images, return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        return {key: value.to(self.device) for key, value in inputs.items()}


def image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


class OpenAICaptioner:
    source = "openai"

    def __init__(
        self,
        model: str,
        *,
        max_output_tokens: int,
        image_detail: str,
    ) -> None:
        from openai import OpenAI

        self.model_path = model
        self.max_output_tokens = max_output_tokens
        self.image_detail = image_detail
        self.client = OpenAI()

    def caption(self, image_paths: list[str], prompt: str) -> str:
        if not image_paths:
            return ""
        content = [{"type": "input_text", "text": prompt}]
        content.extend(
            {
                "type": "input_image",
                "image_url": image_data_url(path),
                "detail": self.image_detail,
            }
            for path in image_paths
        )
        response = self.client.responses.create(
            model=self.model_path,
            input=[{"role": "user", "content": content}],
            max_output_tokens=self.max_output_tokens,
        )
        return clean_caption(response.output_text)


def task_prompt(task: str, caption: str) -> str:
    prefix = TASK_PREFIX.get(task, "Generate an anime animation shot matching this content:")
    return f"{prefix} {caption}".strip()


def caption_records(
    records: list[dict[str, Any]],
    *,
    captioner: Captioner,
    num_frames: int,
    prompt: str,
    overwrite: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    grouped = group_by_shot(records)
    captions: dict[str, tuple[str, list[str]]] = {}
    items = list(grouped.items())
    if limit is not None:
        items = items[:limit]
    for key, shot_records in tqdm(items, desc="caption-anita-shots"):
        existing = next(
            (record.get("caption") for record in shot_records if record.get("caption")),
            None,
        )
        if existing and not overwrite:
            captions[key] = (str(existing), [])
            continue
        paths = representative_paths(best_visual_paths(shot_records), num_frames=num_frames)
        caption = captioner.caption(paths, prompt)
        captions[key] = (caption, paths)

    output = []
    for record in records:
        updated = dict(record)
        key = record_shot_key(record)
        if key in captions:
            caption, paths = captions[key]
            if caption and (overwrite or not updated.get("caption")):
                updated["caption"] = caption
                updated["prompt"] = task_prompt(str(updated.get("task", "")), caption)
                updated["vlm_caption_frame_paths"] = paths
                updated["vlm_caption_source"] = captioner.source
                updated["vlm_caption_model"] = captioner.model_path
        output.append(updated)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Caption Anita production shots with a VLM.")
    parser.add_argument(
        "--input",
        required=True,
        help="Anita production manifest JSONL or Anita root.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=("openai", "local"), default="openai")
    parser.add_argument("--model", default="gpt-5.6")
    parser.add_argument(
        "--model-path",
        help="Local Hugging Face VLM directory for --provider local.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def build_captioner(args: argparse.Namespace) -> Captioner:
    if args.provider == "openai":
        return OpenAICaptioner(
            args.model,
            max_output_tokens=args.max_new_tokens,
            image_detail=args.image_detail,
        )
    if not args.model_path:
        raise ValueError("--model-path is required when --provider local")
    return LocalVLMCaptioner(
        args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
    )


def main() -> None:
    args = build_parser().parse_args()
    records = load_records(args.input)
    captioner = build_captioner(args)
    updated = caption_records(
        records,
        captioner=captioner,
        num_frames=args.num_frames,
        prompt=args.prompt,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    write_jsonl(args.output, updated)
    print(json.dumps({"records": len(updated), "output": args.output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
