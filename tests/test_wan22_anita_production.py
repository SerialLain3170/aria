from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from text_to_anime.train_wan22_anita_production_lora import (
    WanAnitaProductionDataset,
    collate_batch,
    build_task_sample_weights,
    prepare_video_condition,
    prompt_for_record,
)


def _write_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 16), color).save(path)


def _write_scene(root: Path):
    for frame in range(1, 6):
        name = f"{frame:04d}.png"
        _write_image(root / "Hero" / "scene_001" / "sketch" / name, (255, 255, 255))
        _write_image(root / "Hero" / "scene_001" / "color" / name, (255, 0, 0))
        _write_image(root / "Hero" / "scene_001" / "composition" / name, (0, 0, 255))


def test_wan_anita_production_dataset_uses_source_shot_for_stage_two(tmp_path):
    _write_scene(tmp_path)
    dataset = WanAnitaProductionDataset(
        tmp_path,
        num_frames=5,
        height=32,
        width=32,
        random_clip=False,
        random_crop=False,
        first_frame_prob=0.0,
        reference_prob=0.0,
        condition_dropout_prob=0.0,
        text_drop_prob=0.0,
        tasks=("character_color",),
    )

    item = dataset[0]
    batch = collate_batch([item])

    assert item["video"].shape == (3, 5, 32, 32)
    assert item["condition_video"].shape == (3, 5, 32, 32)
    assert item["condition_mask"].shape == (1, 5, 32, 32)
    assert item["source_kind"] == "source_shot"
    assert item["condition_mask"].min().item() == 1.0
    assert batch["video"].shape == (1, 3, 5, 32, 32)


def test_wan_anita_production_dataset_handles_mixed_native_frame_sizes(tmp_path):
    for frame, size in enumerate([(24, 16), (32, 20), (28, 18), (30, 22), (26, 24)], 1):
        name = f"{frame:04d}.png"
        for stage, color in (
            ("sketch", (255, 255, 255)),
            ("color", (255, 0, 0)),
            ("composition", (0, 0, 255)),
        ):
            path = tmp_path / "Hero" / "scene_001" / stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", size, color).save(path)

    dataset = WanAnitaProductionDataset(
        tmp_path,
        num_frames=5,
        height=32,
        width=32,
        random_clip=False,
        random_crop=False,
        first_frame_prob=0.0,
        reference_prob=0.0,
        condition_dropout_prob=0.0,
        text_drop_prob=0.0,
        tasks=("character_color",),
    )

    item = dataset[0]

    assert item["video"].shape == (3, 5, 32, 32)
    assert item["condition_video"].shape == (3, 5, 32, 32)
    assert item["condition_mask"].shape == (1, 5, 32, 32)


def test_prompt_for_record_combines_task_prompt_and_caption():
    prompt = prompt_for_record(
        {
            "task": "compose_refine",
            "prompt": "Complete the shot.",
            "caption": "A girl stands in a classroom.",
        }
    )

    assert "<TASK_COMPOSE_REFINE>" in prompt
    assert "Complete the shot." in prompt
    assert "A girl stands in a classroom." in prompt


class _LatentDist:
    def __init__(self, latents):
        self._latents = latents

    def mode(self):
        return self._latents


class _FakeVAE:
    dtype = torch.float32

    def __init__(self):
        self.config = SimpleNamespace(
            latents_mean=[0.0, 0.0],
            latents_std=[1.0, 1.0],
            z_dim=2,
        )

    def encode(self, video):
        batch = video.shape[0]
        latents = torch.zeros(batch, 2, 2, 4, 4, device=video.device)
        return SimpleNamespace(latent_dist=_LatentDist(latents))


def test_prepare_video_condition_packs_wan_mask_and_latents():
    condition_video = torch.zeros(1, 3, 5, 32, 32)
    condition_mask = torch.zeros(1, 1, 5, 32, 32)
    condition_mask[:, :, 0] = 1.0

    condition = prepare_video_condition(
        vae=_FakeVAE(),
        condition_video=condition_video,
        condition_mask=condition_mask,
        dtype=torch.float32,
        vae_sample_mode="mean",
        vae_scale_factor_temporal=4,
        latent_height=4,
        latent_width=4,
    )

    assert condition.shape == (1, 6, 2, 4, 4)
    assert condition[:, :4, 0].sum().item() == 4 * 4 * 4
    assert condition[:, :4, 1].sum().item() == 0


def test_task_sample_weights_balance_task_probability():
    records = ([{"task": "line_art"}] * 4) + ([{"task": "character_color"}] * 2) + ([{"task": "compose_refine"}] * 1)

    weights = build_task_sample_weights(
        records,
        tasks=("line_art", "character_color", "compose_refine"),
    )

    totals = {"line_art": 0.0, "character_color": 0.0, "compose_refine": 0.0}
    for record, weight in zip(records, weights):
        totals[record["task"]] += weight

    assert totals["line_art"] == totals["character_color"]
    assert totals["character_color"] == totals["compose_refine"]


def test_task_sample_weights_accept_custom_task_mixture():
    records = ([{"task": "line_art"}] * 4) + ([{"task": "character_color"}] * 2) + ([{"task": "compose_refine"}] * 1)

    weights = build_task_sample_weights(
        records,
        tasks=("line_art", "character_color", "compose_refine"),
        task_weights={"compose_refine": 2.0},
    )

    totals = {"line_art": 0.0, "character_color": 0.0, "compose_refine": 0.0}
    for record, weight in zip(records, weights):
        totals[record["task"]] += weight

    assert totals["line_art"] == totals["character_color"]
    assert totals["compose_refine"] == 2.0 * totals["line_art"]
