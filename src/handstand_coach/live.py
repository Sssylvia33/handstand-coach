"""Live Handstand Coach application."""

from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from handstand_coach.capture import OpenCVVideoSource
from handstand_coach.metrics import PoseMetrics, SelectedJointAngle
from handstand_coach.recording import SessionWriter
from handstand_coach.session import create_session_metadata
from handstand_coach.stream import AnalyzedFrame, analyze_stream
from handstand_coach.tracking import create_pose_metrics_tracker
from handstand_coach.ultralytics_estimator import UltralyticsPoseEstimator
from handstand_coach.visualization import PoseRenderer

WINDOW_NAME = "Handstand Coach"


def run_live(
    *,
    source: int | str | Path,
    model_path: str | Path,
    confidence_threshold: float = 0.5,
    record_session: bool = False,
    output_directory: Path = Path("sessions"),
) -> None:
    """Run live pose estimation until the source ends or the user quits."""

    print(f"Loading pose model: {model_path}")
    estimator = UltralyticsPoseEstimator(model_path)
    renderer = PoseRenderer(confidence_threshold=confidence_threshold)

    metrics_tracker = create_pose_metrics_tracker(
        confidence_threshold=confidence_threshold, smoothing_time_constant_s=0.14
    )

    print(f"Opening video source: {source!r}")

    writer: SessionWriter | None = None

    try:
        with ExitStack() as stack:
            video_source = stack.enter_context(OpenCVVideoSource(source))

            if record_session:
                metadata = create_session_metadata(
                    source=source,
                    model=model_path,
                    confidence_threshold=confidence_threshold,
                    started_at=datetime.now(UTC),
                )
                writer = stack.enter_context(
                    SessionWriter(
                        output_directory=output_directory,
                        metadata=metadata,
                    )
                )
                print(f"Recording session to: {writer.session_directory}")

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            for result in analyze_stream(
                video_source,
                estimator,
            ):
                if writer is not None:
                    writer.write_frame(result.pose_frame)

                displayed_image = renderer.render(
                    result.image,
                    result.pose_frame,
                )
                pose_metrics = metrics_tracker.update(result.pose_frame)

                _draw_live_status(
                    displayed_image,
                    result,
                    pose_metrics=pose_metrics,
                )

                cv2.imshow(WINDOW_NAME, displayed_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quit requested")
                    break

                if (
                    cv2.getWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_VISIBLE,
                    )
                    < 1
                ):
                    print("Window closed")
                    break
    finally:
        cv2.destroyAllWindows()

    if writer is not None:
        print(f"Session saved to: {writer.session_directory}")

    print("Live session finished; resources released")


def _format_selected_angle(
    label: str,
    selected_angle: SelectedJointAngle | None,
) -> str:
    """Format one confidence-selected joint angle for the live display."""

    if selected_angle is None:
        return f"{label}: unavailable"

    return f"{label} ({selected_angle.source_side.value}): {selected_angle.angle.degrees:.0f} deg"


def _draw_live_status(
    image: NDArray[np.uint8],
    result: AnalyzedFrame,
    *,
    pose_metrics: PoseMetrics,
) -> None:
    """Draw live status information on an annotated image."""

    pose_detected = pose_metrics.pose_detected
    status = "Pose detected" if pose_detected else "No pose detected"
    status_color = (0, 255, 0) if pose_detected else (0, 0, 255)

    white = (255, 255, 255)
    lines = (
        (status, status_color),
        (f"Processing FPS: {result.processing_fps:.1f}", white),
        (_format_selected_angle("Elbow", pose_metrics.elbow_angle), white),
        (_format_selected_angle("Hip", pose_metrics.hip_angle), white),
        ("Press q to quit", white),
    )

    for index, (text, color) in enumerate(lines):
        cv2.putText(
            image,
            text,
            (20, 30 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
