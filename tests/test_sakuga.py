from text_to_anime.sakuga import sakuga_row_to_manifest


def test_sakuga_row_to_manifest_uses_existing_caption_and_relative_path(tmp_path):
    row = {
        "clip_id": "clip-001",
        "file_path": "aa/bb/clip-001.mp4",
        "video_caption": "A girl blinks and looks away in a classroom.",
        "video_id": "source-1",
        "width": 640,
        "height": 360,
        "seconds": 4.0,
        "tags": ["1girl", "school_uniform"],
    }

    record = sakuga_row_to_manifest(row, clips_root=tmp_path, index=0)

    assert record["caption"] == "A girl blinks and looks away in a classroom."
    assert record["video_path"].endswith("aa/bb/clip-001.mp4")
    assert record["source_id"] == "source-1"
    assert record["anime_tags"] == "1girl, school_uniform"
