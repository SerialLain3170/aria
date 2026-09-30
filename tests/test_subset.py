from text_to_anime.subset import infer_category, select_subset


def test_infer_category_prefers_talking_scene_keywords():
    record = {"caption": "A student quietly speaks while blinking in a close-up shot."}

    assert infer_category(record) == "talking_facial_acting"


def test_select_subset_uses_dataset_caption_without_recaptioning():
    records = [
        {
            "video_path": f"clip-{idx}.mp4",
            "caption": "A girl speaks softly in a classroom close-up.",
            "source_id": f"source-{idx}",
            "width": 640,
            "height": 360,
            "seconds": 4.0,
            "aesthetic_score": 6.0,
            "motion_amplitude": 2,
        }
        for idx in range(25)
    ]

    subset = select_subset(records, target_size=20, min_quality=1.0, max_per_source=2)

    assert len(subset) == 20
    assert all("speaks softly" in record["caption"] for record in subset)
