from text_to_anime.animeshooter import animeshooter_to_records


def test_animeshooter_to_records_uses_native_visual_captions(tmp_path):
    annotation = {
        "video ID": "abc123",
        "fps": 24,
        "segments": [
            {
                "start frame index": 48,
                "story script": {
                    "storyline": "A student waits in the classroom.",
                    "main characters": [
                        {"ID": "c1", "appearance": "a schoolgirl with short black hair"}
                    ],
                    "main scenes": [{"ID": "s1", "environment": "an empty classroom at sunset"}],
                    "shots": [
                        {
                            "start time": 1.0,
                            "end time": 4.0,
                            "main characters": ["c1"],
                            "scene": "s1",
                            "visual annotation": {
                                "narrative caption": "The girl quietly speaks while looking down.",
                                "descriptive caption": "A medium close-up with soft sunset light.",
                            },
                        }
                    ],
                },
            }
        ],
    }

    records = animeshooter_to_records(
        annotation,
        source_name="abc123.json",
        clips_root=tmp_path / "clips",
        videos_root=tmp_path / "videos",
        clip_ext=".mp4",
    )

    assert len(records) == 1
    record = records[0]
    assert "quietly speaks" in record["caption"]
    assert "medium close-up" in record["caption"]
    assert record["video_path"].endswith("abc123/seg000_shot000.mp4")
    assert record["source_video_path"].endswith("abc123.mp4")
    assert record["start_time"] == 3.0
    assert record["end_time"] == 6.0
    assert record["appearance"] == "a schoolgirl with short black hair"
    assert record["background"] == "an empty classroom at sunset"
