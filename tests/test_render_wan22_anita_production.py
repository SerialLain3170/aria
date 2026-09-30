import torch
from test_wan22_anita_production import _write_scene

from text_to_anime.render_wan22_anita_production import load_teacher_forced_sample, video_metrics
from text_to_anime.train_wan22_anita_production_lora import load_records


def test_load_records_filters_by_scene_or_parent_scene(tmp_path):
    records = [
        {"task": "line_art", "scene": "119_a_part000", "parent_scene": "119_a"},
        {"task": "line_art", "scene": "119_a_part001", "parent_scene": "119_a"},
        {"task": "line_art", "scene": "221_a_part000", "parent_scene": "221_a"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(__import__("json").dumps(record) for record in records) + "\n")

    by_scene = load_records(manifest, tasks=("line_art",), include_scenes=("119_a_part001",))
    by_parent = load_records(manifest, tasks=("line_art",), include_scenes=("119_a",))

    assert [record["scene"] for record in by_scene] == ["119_a_part001"]
    assert [record["scene"] for record in by_parent] == ["119_a_part000", "119_a_part001"]


def test_teacher_forced_sample_matches_training_conditioning(tmp_path):
    _write_scene(tmp_path)
    records = load_records(tmp_path, tasks=("line_art", "character_color"))
    by_task = {record["task"]: record for record in records}

    line_art = load_teacher_forced_sample(
        by_task["line_art"], num_frames=5, height=32, width=32, line_art_condition="first_frame"
    )
    color = load_teacher_forced_sample(
        by_task["character_color"], num_frames=5, height=32, width=32, line_art_condition="first_frame"
    )

    assert line_art["source_kind"] == "first_frame"
    assert line_art["mask"][0].min().item() == 255
    assert line_art["mask"][1:].max().item() == 0
    assert color["source_kind"] == "source_shot"
    assert color["mask"].min().item() == 255
    assert color["condition"].shape == color["target"].shape == (5, 3, 32, 32)
    assert color["condition"].dtype == color["target"].dtype == torch.uint8


def test_video_metrics_expose_input_copying():
    condition = torch.full((5, 3, 8, 8), 255, dtype=torch.uint8)
    target = torch.zeros((5, 3, 8, 8), dtype=torch.uint8)
    target[:, 0] = 255

    metrics = video_metrics(condition.clone(), target, condition, "source_shot")

    assert metrics["psnr_gen_vs_condition"] == float("inf")
    assert metrics["psnr_gen_vs_target"] == metrics["psnr_condition_vs_target"]
    assert metrics["saturation_target"] == 1.0
