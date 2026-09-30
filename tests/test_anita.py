from text_to_anime.anita import build_records


def test_anita_build_records_from_sequence_folders(tmp_path):
    seq = tmp_path / "Hero" / "color" / "scene_001"
    seq.mkdir(parents=True)
    (seq / "0001.png").write_bytes(b"x")
    (seq / "0002.png").write_bytes(b"x")

    records = build_records(tmp_path, clips_root=tmp_path / "clips", fps=12)

    assert len(records) == 1
    record = records[0]
    assert record["dataset"] == "AnitaDataset"
    assert record["work"] == "Hero"
    assert record["scene"] == "scene_001"
    assert record["sequence_type"] == "color"
    assert "segmented color" in record["caption"]
    assert record["video_path"].endswith("Hero/scene_001_color.mp4")
