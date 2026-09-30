from text_to_anime.caption_anita import (
    OpenAICaptioner,
    best_visual_paths,
    caption_records,
    group_by_shot,
    representative_paths,
)


class FakeCaptioner:
    model_path = "/models/local-vlm"

    def caption(self, image_paths, prompt):
        assert prompt
        assert image_paths
        return "A girl turns toward the camera in a sunlit classroom."


def _records():
    return [
        {
            "task": "line_art",
            "work": "Hero",
            "scene": "scene_001",
            "target_paths": ["sketch/0001.png", "sketch/0002.png", "sketch/0003.png"],
        },
        {
            "task": "character_color",
            "work": "Hero",
            "scene": "scene_001",
            "target_paths": ["color/0001.png", "color/0002.png", "color/0003.png"],
        },
        {
            "task": "compose_refine",
            "work": "Hero",
            "scene": "scene_001",
            "target_paths": [
                "composition/0001.png",
                "composition/0002.png",
                "composition/0003.png",
            ],
        },
    ]


def test_best_visual_paths_prefers_composition_targets():
    paths = best_visual_paths(_records())

    assert paths[0] == "composition/0001.png"


def test_representative_paths_are_unique_and_centered():
    paths = representative_paths(["a", "b", "c", "d"], num_frames=3)

    assert paths[0] == "a"
    assert len(paths) == 3
    assert len(set(paths)) == 3


def test_caption_records_applies_one_caption_to_all_task_records():
    updated = caption_records(
        _records(),
        captioner=FakeCaptioner(),
        num_frames=2,
        prompt="Caption this shot.",
        overwrite=False,
        limit=None,
    )

    assert {record["caption"] for record in updated} == {
        "A girl turns toward the camera in a sunlit classroom."
    }
    assert all("sunlit classroom" in record["prompt"] for record in updated)
    assert all(record["vlm_caption_model"] == "/models/local-vlm" for record in updated)


def test_openai_captioner_uses_responses_api_without_network(monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Response:
                output_text = "A quiet anime classroom shot."

            return Response()

    class FakeClient:
        responses = FakeResponses()

    def fake_init(self, model, *, max_output_tokens, image_detail):
        self.model_path = model
        self.max_output_tokens = max_output_tokens
        self.image_detail = image_detail
        self.client = FakeClient()

    monkeypatch.setattr(OpenAICaptioner, "__init__", fake_init)
    captioner = OpenAICaptioner("gpt-5.6", max_output_tokens=80, image_detail="low")

    caption = captioner.caption([str(image)], "Describe this shot.")

    assert caption == "A quiet anime classroom shot."
    assert captured["model"] == "gpt-5.6"
    content = captured["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
