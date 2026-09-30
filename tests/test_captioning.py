from text_to_anime.captioning import assess_caption, build_caption


def test_build_caption_from_structured_fields():
    record = {
        "subject": "a high-school girl",
        "appearance": "short black hair and a navy uniform",
        "initial_state": "standing beside a classroom window",
        "action": "lowers her eyes and quietly begins speaking",
        "secondary_motion": "her hair moves slightly in the evening breeze",
        "camera": "slow push-in from medium shot to close-up",
        "background": "an empty classroom at sunset",
        "style": "hand-drawn Japanese cel animation",
        "motion_amplitude": 2,
    }

    caption = build_caption(record)

    assert "high-school girl" in caption
    assert "slow push-in" in caption
    assert "motion score: 2.0" in caption
    assert "There is no text in the video." in caption


def test_assess_caption_allows_existing_caption():
    quality = assess_caption({"caption": "A quiet close-up of a student blinking."})

    assert quality.caption.startswith("A quiet close-up")
    assert quality.ok is False

