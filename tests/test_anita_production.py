from pathlib import Path

import torch
from PIL import Image

from text_to_anime.anita import build_records
from text_to_anime.anita_production import (
    AnitaProductionDataset,
    AnitaProductionUNet3D,
    build_production_samples,
    iter_shot_sets,
)


def _write_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (24, 16), color).save(path)


def _write_scene(root: Path):
    for frame in range(1, 5):
        name = f"{frame:04d}.png"
        _write_image(root / "Hero" / "scene_001" / "sketch" / name, (255, 255, 255, 255))
        _write_image(root / "Hero" / "scene_001" / "color" / name, (255, 0, 0, 255))
        _write_image(root / "Hero" / "scene_001" / "composition" / name, (0, 0, 255, 255))


def test_anita_manifest_supports_official_scene_layout(tmp_path):
    _write_scene(tmp_path)

    records = build_records(tmp_path, clips_root=tmp_path / "clips", fps=12)

    assert {record["sequence_type"] for record in records} == {
        "sketch",
        "color",
        "composition",
    }
    assert {record["scene"] for record in records} == {"scene_001"}


def test_production_samples_support_extracted_anita_layout(tmp_path):
    for frame in range(1, 3):
        name = f"{frame:04d}.png"
        _write_image(tmp_path / "Hero" / "sketch" / "scene_001" / name, (255, 255, 255, 255))
        _write_image(tmp_path / "Hero" / "color" / "scene_001" / name, (255, 0, 0, 255))
        _write_image(tmp_path / "Hero" / "composition" / "scene_001" / name, (0, 0, 255, 255))

    samples = build_production_samples(tmp_path)

    assert {sample["task"] for sample in samples} == {
        "line_art",
        "character_color",
        "compose_refine",
    }
    character = next(sample for sample in samples if sample["task"] == "character_color")
    assert character["primary_paths"][0].endswith("sketch/scene_001/0001.png")
    assert character["target_paths"][0].endswith("color/scene_001/0001.png")


def test_production_samples_split_long_scenes_into_subshots(tmp_path):
    for frame in range(1, 8):
        name = f"{frame:04d}.png"
        _write_image(tmp_path / "Hero" / "sketch" / "scene_001" / name, (255, 255, 255, 255))

    samples = build_production_samples(
        tmp_path,
        tasks=("line_art",),
        max_frames_per_shot=3,
        min_frames_per_shot=2,
        shot_split_strategy="frame_count",
    )

    assert [sample["scene"] for sample in samples] == [
        "scene_001_part000",
        "scene_001_part001",
        "scene_001_part002",
    ]
    assert [sample["parent_scene"] for sample in samples] == ["scene_001"] * 3
    assert [sample["subshot_index"] for sample in samples] == [0, 1, 2]
    assert [sample["num_subshots"] for sample in samples] == [3, 3, 3]
    assert [sample["frame_ids"] for sample in samples] == [
        ["0001", "0002", "0003"],
        ["0004", "0005"],
        ["0006", "0007"],
    ]


def test_production_samples_split_at_semantic_transition(tmp_path):
    for frame in range(1, 11):
        name = f"{frame:04d}.png"
        color = (0, 0, 0, 255) if frame <= 4 else (255, 255, 255, 255)
        _write_image(tmp_path / "Hero" / "sketch" / "scene_001" / name, color)

    samples = build_production_samples(
        tmp_path,
        tasks=("line_art",),
        max_frames_per_shot=6,
        min_frames_per_shot=2,
        shot_split_strategy="semantic",
    )

    assert [sample["frame_ids"] for sample in samples] == [
        ["0001", "0002", "0003", "0004"],
        ["0005", "0006", "0007", "0008", "0009", "0010"],
    ]
    assert {sample["split_strategy"] for sample in samples} == {"semantic"}


def test_build_production_samples_from_paired_shot_dirs(tmp_path):
    _write_scene(tmp_path)

    shot_sets = list(iter_shot_sets(tmp_path))
    samples = build_production_samples(tmp_path)

    assert len(shot_sets) == 1
    assert {sample["task"] for sample in samples} == {
        "line_art",
        "character_color",
        "compose_refine",
    }
    character = next(sample for sample in samples if sample["task"] == "character_color")
    compose = next(sample for sample in samples if sample["task"] == "compose_refine")
    assert len(character["primary_paths"]) == 4
    assert character["primary_paths"][0].endswith("sketch/0001.png")
    assert compose["primary_paths"][0].endswith("color/0001.png")
    assert compose["target_paths"][0].endswith("composition/0001.png")


def test_production_dataset_and_3d_model_forward(tmp_path):
    _write_scene(tmp_path)
    dataset = AnitaProductionDataset(
        tmp_path,
        num_frames=3,
        height=32,
        width=32,
        random_clip=False,
        random_crop=False,
        first_frame_prob=1.0,
        reference_prob=1.0,
        text_drop_prob=0.0,
    )
    item = dataset[0]

    assert item["condition"].shape == (12, 3, 32, 32)
    assert item["target"].shape == (3, 3, 32, 32)

    batch_condition = item["condition"].unsqueeze(0)
    model = AnitaProductionUNet3D(
        base_channels=8,
        channel_mults=(1, 2),
        cond_dim=16,
        vocab_size=128,
    )
    output = model(batch_condition, [item["task"]], [item["prompt"]])

    assert output.shape == (1, 3, 3, 32, 32)
    assert torch.isfinite(output).all()
