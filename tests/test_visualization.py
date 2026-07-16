import numpy as np
import pytest

from handstand_coach.models import Keypoint, KeypointName, Pose, PoseFrame
from handstand_coach.visualization import PoseRenderer


def make_pose_frame(
    *keypoints: Keypoint,
    image_width: int = 100,
    image_height: int = 100,
) -> PoseFrame:
    pose = Pose(keypoints=keypoints) if keypoints else None

    return PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_width=image_width,
        image_height=image_height,
        pose=pose,
    )


def make_keypoint(
    name: KeypointName,
    *,
    x: float,
    y: float,
    confidence: float = 0.9,
) -> Keypoint:
    return Keypoint(
        name=name,
        x=x,
        y=y,
        confidence=confidence,
    )


def test_to_pixel_maps_normalized_boundaries_to_valid_pixels() -> None:
    top_left = make_keypoint(
        KeypointName.NOSE,
        x=0.0,
        y=0.0,
    )
    bottom_right = make_keypoint(
        KeypointName.NOSE,
        x=1.0,
        y=1.0,
    )

    assert PoseRenderer._to_pixel(top_left, 640, 480) == (0, 0)
    assert PoseRenderer._to_pixel(bottom_right, 640, 480) == (639, 479)


def test_render_returns_unchanged_copy_when_no_pose_exists() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    pose_frame = make_pose_frame()
    renderer = PoseRenderer()

    rendered = renderer.render(image, pose_frame)

    assert rendered is not image
    assert np.array_equal(rendered, image)


def test_render_draws_connection_between_reliable_keypoints() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    left_shoulder = make_keypoint(
        KeypointName.LEFT_SHOULDER,
        x=0.2,
        y=0.5,
    )
    right_shoulder = make_keypoint(
        KeypointName.RIGHT_SHOULDER,
        x=0.8,
        y=0.5,
    )
    pose_frame = make_pose_frame(left_shoulder, right_shoulder)
    renderer = PoseRenderer(
        confidence_threshold=0.5,
        connection_color=(255, 255, 255),
        keypoint_color=(255, 255, 255),
        connection_thickness=1,
        keypoint_radius=2,
    )

    rendered = renderer.render(image, pose_frame)

    # The midpoint between the two shoulders lies on the connection.
    assert np.any(rendered[50, 50] > 0)

    # Rendering must not modify the source image.
    assert not np.any(image)


def test_render_omits_connection_when_endpoint_has_low_confidence() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    left_shoulder = make_keypoint(
        KeypointName.LEFT_SHOULDER,
        x=0.2,
        y=0.5,
        confidence=0.9,
    )
    right_shoulder = make_keypoint(
        KeypointName.RIGHT_SHOULDER,
        x=0.8,
        y=0.5,
        confidence=0.4,
    )
    pose_frame = make_pose_frame(left_shoulder, right_shoulder)
    renderer = PoseRenderer(
        confidence_threshold=0.5,
        connection_color=(255, 255, 255),
        keypoint_color=(255, 255, 255),
        connection_thickness=1,
        keypoint_radius=2,
    )

    rendered = renderer.render(image, pose_frame)

    left_pixel = PoseRenderer._to_pixel(left_shoulder, 100, 100)
    right_pixel = PoseRenderer._to_pixel(right_shoulder, 100, 100)

    # Reliable left shoulder is drawn.
    assert np.any(rendered[left_pixel[1], left_pixel[0]] > 0)

    # Unreliable right shoulder is not drawn.
    assert not np.any(rendered[right_pixel[1], right_pixel[0]])

    # No line crosses the midpoint.
    assert not np.any(rendered[50, 50])


def test_render_rejects_image_metadata_mismatch() -> None:
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    pose_frame = make_pose_frame(
        image_width=100,
        image_height=100,
    )
    renderer = PoseRenderer()

    with pytest.raises(ValueError, match="dimensions do not match"):
        renderer.render(image, pose_frame)


@pytest.mark.parametrize(
    "confidence_threshold",
    [
        -0.01,
        1.01,
    ],
)
def test_renderer_rejects_invalid_confidence_threshold(
    confidence_threshold: float,
) -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        PoseRenderer(confidence_threshold=confidence_threshold)
