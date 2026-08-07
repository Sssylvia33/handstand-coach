import pytest

from handstand_coach.geometry import (
    Point2D,
    angle_degrees,
    normalized_to_pixel,
)


def test_normalized_to_pixel_uses_image_dimensions() -> None:
    normalized_point = Point2D(
        x=0.5,
        y=0.25,
    )

    pixel_point = normalized_to_pixel(
        normalized_point,
        image_width=1280,
        image_height=720,
    )

    assert pixel_point == Point2D(
        x=639.5,
        y=179.75,
    )


def test_angle_degrees_returns_right_angle() -> None:
    angle = angle_degrees(
        first=Point2D(x=1.0, y=0.0),
        vertex=Point2D(x=0.0, y=0.0),
        third=Point2D(x=0.0, y=1.0),
    )

    assert angle == pytest.approx(90.0)


def test_angle_degrees_returns_straight_angle() -> None:
    angle = angle_degrees(
        first=Point2D(x=-1.0, y=0.0),
        vertex=Point2D(x=0.0, y=0.0),
        third=Point2D(x=1.0, y=0.0),
    )

    assert angle == pytest.approx(180.0)


def test_angle_degrees_rejects_point_at_vertex() -> None:
    shared_point = Point2D(x=0.0, y=0.0)

    with pytest.raises(
        ValueError,
        match="undefined",
    ):
        angle_degrees(
            first=shared_point,
            vertex=shared_point,
            third=Point2D(x=1.0, y=0.0),
        )


def test_pixel_conversion_preserves_angle_in_non_square_image() -> None:
    image_width = 201
    image_height = 101

    first = normalized_to_pixel(
        Point2D(x=0.75, y=0.9),
        image_width=image_width,
        image_height=image_height,
    )
    vertex = normalized_to_pixel(
        Point2D(x=0.5, y=0.4),
        image_width=image_width,
        image_height=image_height,
    )
    third = normalized_to_pixel(
        Point2D(x=0.25, y=0.9),
        image_width=image_width,
        image_height=image_height,
    )

    angle = angle_degrees(first, vertex, third)

    assert angle == pytest.approx(90.0)


@pytest.mark.parametrize(
    ("image_width", "image_height"),
    [
        (0, 100),
        (-1, 100),
        (100, 0),
        (100, -1),
    ],
)
def test_normalized_to_pixel_rejects_non_positive_dimensions(
    image_width: int,
    image_height: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        normalized_to_pixel(
            Point2D(x=0.5, y=0.5),
            image_width=image_width,
            image_height=image_height,
        )
