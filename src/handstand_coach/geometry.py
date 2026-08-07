"""Two-dimensional geometry for posture analysis."""

from dataclasses import dataclass
from math import acos, degrees, hypot


@dataclass(frozen=True, slots=True)
class Point2D:
    """A point in a two-dimensional coordinate system."""

    x: float
    y: float


def normalized_to_pixel(
    point: Point2D,
    *,
    image_width: int,
    image_height: int,
) -> Point2D:
    """Convert a normalized point to continuous pixel coordinates."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    return Point2D(
        x=point.x * (image_width - 1),
        y=point.y * (image_height - 1),
    )


def angle_degrees(
    first: Point2D,
    vertex: Point2D,
    third: Point2D,
) -> float:
    """Return the smaller angle at the vertex in degrees."""

    first_vector_x = first.x - vertex.x
    first_vector_y = first.y - vertex.y
    third_vector_x = third.x - vertex.x
    third_vector_y = third.y - vertex.y

    first_length = hypot(first_vector_x, first_vector_y)
    third_length = hypot(third_vector_x, third_vector_y)

    if first_length == 0.0 or third_length == 0.0:
        raise ValueError("Angle is undefined when a point coincides with the vertex")

    dot_product = first_vector_x * third_vector_x + first_vector_y * third_vector_y

    cosine = dot_product / (first_length * third_length)
    clamped_cosine = max(-1.0, min(1.0, cosine))

    return degrees(acos(clamped_cosine))
