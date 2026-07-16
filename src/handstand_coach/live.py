"""Live Handstand Coach application."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from handstand_coach.capture import OpenCVVideoSource
from handstand_coach.stream import AnalyzedFrame, analyze_stream
from handstand_coach.ultralytics_estimator import UltralyticsPoseEstimator
from handstand_coach.visualization import PoseRenderer

WINDOW_NAME = "Handstand Coach"


def run_live(
    *,
    source: int | str | Path,
    model_path: str | Path,
    confidence_threshold: float = 0.5,
) -> None:
    """Run live pose estimation until the source ends or the user quits."""

    print(f"Loading pose model: {model_path}")
    estimator = UltralyticsPoseEstimator(model_path)
    renderer = PoseRenderer(confidence_threshold=confidence_threshold)

    print(f"Opening video source: {source!r}")

    try:
        with OpenCVVideoSource(source) as video_source:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            for result in analyze_stream(video_source, estimator):
                displayed_image = renderer.render(
                    result.image,
                    result.pose_frame,
                )
                _draw_live_status(displayed_image, result)

                cv2.imshow(WINDOW_NAME, displayed_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quit requested")
                    break

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed")
                    break
    finally:
        cv2.destroyAllWindows()

    print("Live session finished; resources released")


def _draw_live_status(
    image: NDArray[np.uint8],
    result: AnalyzedFrame,
) -> None:
    """Draw live status information on an annotated image."""

    pose_detected = result.pose_frame.pose is not None
    status = "Pose detected" if pose_detected else "No pose detected"
    status_color = (0, 255, 0) if pose_detected else (0, 0, 255)

    cv2.putText(
        image,
        status,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Processing FPS: {result.processing_fps:.1f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Press q to quit",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
