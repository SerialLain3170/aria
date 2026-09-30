from PIL import Image

from text_to_anime.infer_wan_i2v import resize_crop, rounded_dimensions_for_image


def test_i2v_resize_crop_keeps_requested_size():
    image = Image.new("RGB", (800, 600), "white")

    result = resize_crop(image, height=360, width=640)

    assert result.size == (640, 360)


def test_rounded_dimensions_are_multiple():
    image = Image.new("RGB", (800, 600), "white")

    height, width = rounded_dimensions_for_image(image, max_area=640 * 360, multiple=16)

    assert height % 16 == 0
    assert width % 16 == 0
