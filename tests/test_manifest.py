from text_to_anime.manifest import split_by_source


def test_split_by_source_keeps_matching_sources_together():
    records = [
        {"video_path": "a1.mp4", "source_id": "show-a"},
        {"video_path": "a2.mp4", "source_id": "show-a"},
        {"video_path": "b1.mp4", "source_id": "show-b"},
    ]

    train, val = split_by_source(records, val_ratio=0.5)

    locations = {}
    for record in train:
        locations.setdefault(record["source_id"], set()).add("train")
    for record in val:
        locations.setdefault(record["source_id"], set()).add("val")

    assert all(len(targets) == 1 for targets in locations.values())

