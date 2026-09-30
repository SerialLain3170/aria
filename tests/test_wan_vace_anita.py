import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from text_to_anime.anita_bg_plates import build_scene_plate, push_pull_fill
from text_to_anime.render_wan_vace_anita import mask_to_pil_frames, to_pil_frames
from text_to_anime.train_wan_vace_anita import (
    VaceAnitaDataset,
    build_clip_sources,
    build_sample,
    keyframe_indices,
    latent_weight_map,
    letterbox_like_vace,
    timeline_from_ids,
    wan_frame_count,
)

BG = (40, 120, 200)


def _write_shot(root: Path, frame_ids: list[str], size=(64, 48)):
    for step, frame_id in enumerate(frame_ids):
        x = 4 + (2 * step) % 44
        sketch = Image.new("RGBA", size, (0, 0, 0, 0))
        color = Image.new("RGBA", size, (0, 0, 0, 0))
        for px in range(x, x + 12):
            for py in range(16, 32):
                sketch.putpixel((px, py), (0, 0, 0, 255) if px in (x, x + 11) else (0, 0, 0, 0))
                color.putpixel((px, py), (220, 30, 30, 255))
        composition = Image.new("RGB", size, BG)
        composition.paste(color, (0, 0), color)
        for stage, image in (("sketch", sketch), ("color", color), ("composition", composition)):
            path = root / "work" / stage / "001_a" / f"{frame_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)


def test_timeline_expands_held_drawings():
    assert timeline_from_ids(["0001", "0004", "0006"]) == ["0001"] * 3 + ["0004"] * 2 + ["0006"]
    assert wan_frame_count(40, 33) == 33
    assert wan_frame_count(20, 33) == 17
    assert keyframe_indices(17, 8) == [0, 8, 16]
    assert keyframe_indices(21, 8) == [0, 8, 16, 20]


def test_push_pull_fill_fills_holes_smoothly():
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[:, :8] = 100
    rgb[:, 8:] = 200
    known = np.ones((16, 16), dtype=bool)
    known[4:12, 6:10] = False
    filled = push_pull_fill(rgb, known)
    assert np.allclose(filled[known], rgb[known])
    assert 100 <= filled[8, 7, 0] <= 200


def test_background_plate_removes_moving_character(tmp_path):
    ids = [f"{index:04d}" for index in range(1, 9)]
    _write_shot(tmp_path, ids)
    shot = tmp_path / "work"
    info = build_scene_plate(
        color_paths={i: str(shot / "color" / "001_a" / f"{i}.png") for i in ids},
        composition_paths={i: str(shot / "composition" / "001_a" / f"{i}.png") for i in ids},
        output_dir=tmp_path / "plates",
        max_side=1024,
        alpha_threshold=8,
        dilate=1,
        max_median_frames=48,
        static_threshold=6.0,
    )
    plate = np.asarray(Image.open(info["plate_path"]))
    assert info["static"]
    assert np.abs(plate.astype(int) - np.array(BG)).max() <= 2
    background = np.asarray(Image.open(info["background_paths"]["0001"]))
    assert np.abs(background.astype(int) - np.array(BG)).max() <= 2


def _sources(tmp_path, tasks):
    ids = [f"{index:04d}" for index in range(1, 40, 2)]  # drawings on 2s
    _write_shot(tmp_path, ids)
    shot = tmp_path / "work"
    info = build_scene_plate(
        color_paths={i: str(shot / "color" / "001_a" / f"{i}.png") for i in ids},
        composition_paths={i: str(shot / "composition" / "001_a" / f"{i}.png") for i in ids},
        output_dir=tmp_path / "plates",
        max_side=1024,
        alpha_threshold=8,
        dilate=1,
        max_median_frames=48,
        static_threshold=6.0,
    )
    index = tmp_path / "plates.jsonl"
    index.write_text(__import__("json").dumps({"work": "work", "scene": "001_a", **info}) + "\n")
    return build_clip_sources(tmp_path, tasks=tasks, plates_index=str(index), min_frames=9, timeline_stride=2)


def test_build_sample_poses_each_task_as_vace_conditioning(tmp_path):
    sources = {s["task"]: s for s in _sources(tmp_path, ("inbetween", "character_color", "compose_refine"))}
    assert set(sources) == {"inbetween", "character_color", "compose_refine"}
    kwargs = {"height": 32, "width": 48, "max_frames": 17, "timeline_stride": 2, "keyframe_stride": 8,
              "compose_plate_reference": True, "rng": random.Random(0), "center": True}

    inbetween = build_sample(sources["inbetween"], **kwargs)
    assert inbetween["target"].shape == (17, 3, 32, 48)
    assert inbetween["mask"][[0, 8, 16]].max() == 0 and inbetween["mask"][1].min() == 1
    assert torch.equal(inbetween["control"][8], inbetween["target"][8])
    assert inbetween["control"][1].unique().tolist() == [128]

    color = build_sample(sources["character_color"], **kwargs)
    assert color["mask"].min() == 1 and color["reference"] is not None
    assert color["reference"].size[0] <= 48 and color["reference"].size[1] <= 32

    compose = build_sample(sources["compose_refine"], **kwargs)
    # The rough composite (character over recovered background) already matches this synthetic target.
    assert (compose["control"].int() - compose["target"].int()).abs().float().mean() < 3
    assert compose["reference"] is not None  # static plate


def test_training_conditioning_matches_wan_vace_pipeline_preprocessing(tmp_path):
    from diffusers import WanVACEPipeline
    from diffusers.video_processor import VideoProcessor

    source = _sources(tmp_path, ("character_color",))[0]
    sample = build_sample(source, height=32, width=48, max_frames=17, timeline_stride=2, keyframe_stride=8,
                          compose_plate_reference=True, rng=random.Random(0), center=True)
    fake_pipe = SimpleNamespace(
        video_processor=VideoProcessor(vae_scale_factor=8),
        vae_scale_factor_spatial=8,
        transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
    )
    video, mask, references = WanVACEPipeline.preprocess_conditions(
        fake_pipe,
        video=to_pil_frames(sample["control"]),
        mask=mask_to_pil_frames(sample["mask"]),
        reference_images=sample["reference"],
        height=32, width=48, num_frames=17, dtype=torch.float32, device=torch.device("cpu"),
    )
    train_control = (sample["control"].float() / 127.5 - 1.0).permute(1, 0, 2, 3)
    assert torch.allclose(video[0], train_control, atol=1e-5)
    assert torch.equal(mask[0, :1], sample["mask"].permute(1, 0, 2, 3))
    train_reference = letterbox_like_vace(sample["reference"], height=32, width=48)
    assert torch.allclose(references[0][0], train_reference, atol=1e-5)


def test_dataset_item_and_loss_weights(tmp_path):
    sources = _sources(tmp_path, ("character_color",))
    dataset = VaceAnitaDataset(sources, captions={}, height=32, width=48, max_frames=17, timeline_stride=2,
                               keyframe_stride=8, compose_plate_reference=True, hflip_prob=0.0,
                               text_drop_prob=0.0, deterministic=True)
    item = dataset[0]
    assert item["video"].shape == item["control"].shape == (3, 17, 32, 48)
    assert item["mask"].shape == (1, 17, 32, 48)
    weights = latent_weight_map(item["video"], item["control"], num_reference=1,
                                latent_height=4, latent_width=6, floor=0.2, threshold=0.08)
    assert weights.shape == (1, 1, 1 + 5, 4, 6)
    assert weights[0, 0, 0].max() == 0.2 and weights.max() == 1.0
