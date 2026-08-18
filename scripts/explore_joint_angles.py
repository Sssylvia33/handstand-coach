"""Explore joint-angle quality in a recorded pose session."""

from argparse import ArgumentParser
from math import isfinite, isnan
from pathlib import Path
from statistics import mean, median, pstdev

from matplotlib import pyplot as plt

from handstand_coach.metrics import (
    calculate_joint_angle,
    select_joint_angle,
)
from handstand_coach.models import BodySide, KeypointName
from handstand_coach.reading import SessionReader
from handstand_coach.temporal import ExponentialSmoother


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("session_directory", type=Path)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--end-time",
        type=float,
    )
    parser.add_argument(
        "--joint",
        choices=("elbow", "hip"),
        default="elbow",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right", "best"),
        default="left",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
    )
    parser.add_argument(
        "--reference-angle",
        type=float,
    )
    parser.add_argument(
        "--smoothing-time-constant",
        type=float,
    )

    arguments = parser.parse_args()

    if arguments.start_time < 0.0:
        parser.error("--start-time must be non-negative")

    if arguments.end_time is not None and arguments.end_time <= arguments.start_time:
        parser.error("--end-time must be greater than --start-time")

    if arguments.smoothing_time_constant is not None and (
        not isfinite(arguments.smoothing_time_constant) or arguments.smoothing_time_constant <= 0.0
    ):
        parser.error("--smoothing-time-constant must be a finite positive value")

    joint_keypoints = {
        ("elbow", "left"): (
            KeypointName.LEFT_SHOULDER,
            KeypointName.LEFT_ELBOW,
            KeypointName.LEFT_WRIST,
        ),
        ("elbow", "right"): (
            KeypointName.RIGHT_SHOULDER,
            KeypointName.RIGHT_ELBOW,
            KeypointName.RIGHT_WRIST,
        ),
        ("hip", "left"): (
            KeypointName.LEFT_SHOULDER,
            KeypointName.LEFT_HIP,
            KeypointName.LEFT_KNEE,
        ),
        ("hip", "right"): (
            KeypointName.RIGHT_SHOULDER,
            KeypointName.RIGHT_HIP,
            KeypointName.RIGHT_KNEE,
        ),
    }
    if arguments.side == "best":
        left_names = joint_keypoints[(arguments.joint, "left")]
        right_names = joint_keypoints[(arguments.joint, "right")]
    else:
        selected_names = joint_keypoints[(arguments.joint, arguments.side)]

    reader = SessionReader(arguments.session_directory)
    angles: list[float] = []
    timestamps: list[float] = []
    plotted_angles: list[float] = []
    smoothed_angles: list[float] | None = None

    selected_side_counts = {
        BodySide.LEFT: 0,
        BodySide.RIGHT: 0,
    }
    previous_selected_side: BodySide | None = None
    side_switches = 0

    total_frames = 0

    for pose_frame in reader.iter_frames():
        if pose_frame.timestamp_s < arguments.start_time:
            continue

        if arguments.end_time is not None and pose_frame.timestamp_s >= arguments.end_time:
            break

        total_frames += 1
        timestamps.append(pose_frame.timestamp_s)

        if arguments.side == "best":
            left_result = calculate_joint_angle(
                pose_frame,
                first_name=left_names[0],
                vertex_name=left_names[1],
                third_name=left_names[2],
                confidence_threshold=arguments.confidence_threshold,
            )
            right_result = calculate_joint_angle(
                pose_frame,
                first_name=right_names[0],
                vertex_name=right_names[1],
                third_name=right_names[2],
                confidence_threshold=arguments.confidence_threshold,
            )

            selected = select_joint_angle(
                left=left_result,
                right=right_result,
            )
            if selected is None:
                result = None
            else:
                result = selected.angle
                selected_side_counts[selected.source_side] += 1

                if (
                    previous_selected_side is not None
                    and selected.source_side is not previous_selected_side
                ):
                    side_switches += 1

                previous_selected_side = selected.source_side
        else:
            result = calculate_joint_angle(
                pose_frame,
                first_name=selected_names[0],
                vertex_name=selected_names[1],
                third_name=selected_names[2],
                confidence_threshold=arguments.confidence_threshold,
            )

        if result is None:
            plotted_angles.append(float("nan"))
        else:
            angles.append(result.degrees)
            plotted_angles.append(result.degrees)

    if arguments.smoothing_time_constant is not None:
        smoother = ExponentialSmoother(time_constant_s=arguments.smoothing_time_constant)
        smoothed_angles = []

        for timestamp_s, angle in zip(
            timestamps,
            plotted_angles,
            strict=True,
        ):
            if isnan(angle):
                smoother.reset()
                smoothed_angles.append(angle)
                continue

            smoothed_angles.append(
                smoother.update(
                    value=angle,
                    timestamp_s=timestamp_s,
                )
            )
    usable_frames = len(angles)
    usable_percentage = 100.0 * usable_frames / total_frames if total_frames else 0.0

    print(f"Total frames: {total_frames}")
    print(
        f"Usable {arguments.side}-{arguments.joint} angles: "
        f"{usable_frames}/{total_frames} "
        f"({usable_percentage:.1f}%)"
    )

    if not angles:
        print("No usable angles were calculated")
        return 0

    print(f"Mean angle: {mean(angles):.1f} degrees")
    print(f"Median angle: {median(angles):.1f} degrees")
    print(f"Minimum angle: {min(angles):.1f} degrees")
    print(f"Maximum angle: {max(angles):.1f} degrees")
    if len(angles) < 2:
        print("Standard deviation: unavailable; insufficient samples")
    else:
        print(f"Standard deviation: {pstdev(angles):.1f} degrees")

    if arguments.side == "best":
        print(f"Selected left-labelled frames: {selected_side_counts[BodySide.LEFT]}")
        print(f"Selected right-labelled frames: {selected_side_counts[BodySide.RIGHT]}")
        print(f"Side switches: {side_switches}")

    if arguments.plot_output is not None:
        figure, axis = plt.subplots(figsize=(10, 5))

        axis.plot(
            timestamps,
            plotted_angles,
            marker=".",
            markersize=4,
            linewidth=1,
            label=f"{arguments.side} {arguments.joint}",
        )
        if smoothed_angles is not None:
            axis.plot(
                timestamps,
                smoothed_angles,
                linewidth=2,
                label=(f"EMA time constant={arguments.smoothing_time_constant:.2f}s"),
            )

        if arguments.reference_angle is not None:
            axis.axhline(
                arguments.reference_angle,
                linestyle="--",
                linewidth=1.5,
                label=f"Reference: {arguments.reference_angle:.0f} degrees",
            )

        axis.set_title(f"{arguments.side.title()} {arguments.joint} angle over time")
        axis.set_xlabel("Session time (seconds)")
        axis.set_ylabel(f"{arguments.joint.title()} angle (degrees)")
        axis.set_ylim(0.0, 185.0)
        axis.grid(alpha=0.3)
        axis.legend()

        figure.tight_layout()
        arguments.plot_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        figure.savefig(
            arguments.plot_output,
            dpi=150,
        )
        plt.close(figure)

        print(f"Plot saved to: {arguments.plot_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
