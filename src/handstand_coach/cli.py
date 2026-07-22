"""Command-line interface for Handstand Coach."""

from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from pathlib import Path

from handstand_coach.capture import VideoSourceError
from handstand_coach.estimation import PoseModelLoadError
from handstand_coach.live import run_live
from handstand_coach.recording import SessionWriteError


def parse_source(value: str) -> int | str:
    """Convert numeric sources to camera indices; preserve other sources."""

    if value.isdecimal():
        return int(value)

    return value


def parse_confidence_threshold(value: str) -> float:
    """Parse and validate a normalized confidence threshold."""

    try:
        threshold = float(value)
    except ValueError as error:
        raise ArgumentTypeError("must be a number") from error

    if not 0.0 <= threshold <= 1.0:
        raise ArgumentTypeError("must be between 0.0 and 1.0")

    return threshold


def build_parser() -> ArgumentParser:
    """Build the Handstand Coach argument parser."""

    parser = ArgumentParser(
        prog="handstand-coach",
        description="Real-time handstand posture analysis.",
    )
    subcommands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    live_parser = subcommands.add_parser(
        "live",
        help="Analyze a live camera or video source.",
    )
    live_parser.add_argument(
        "--source",
        type=parse_source,
        default=0,
        help="Camera index, video path, or stream URL. Default: 0",
    )
    live_parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolov8n-pose.pt"),
        help="Path or name of an Ultralytics pose model.",
    )
    live_parser.add_argument(
        "--confidence-threshold",
        type=parse_confidence_threshold,
        default=0.5,
        help="Minimum keypoint confidence from 0.0 to 1.0. Default: 0.5",
    )
    live_parser.add_argument(
        "--record-session",
        action="store_true",
        help="Save structured pose data for this session.",
    )
    live_parser.add_argument(
        "--output-dir",
        dest="output_directory",
        type=Path,
        default=Path("sessions"),
        help="Directory used for recorded sessions. Default: sessions",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line application."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "live":
        try:
            run_live(
                source=arguments.source,
                model_path=arguments.model,
                confidence_threshold=arguments.confidence_threshold,
                record_session=arguments.record_session,
                output_directory=arguments.output_directory,
            )
        except VideoSourceError as error:
            parser.exit(
                status=1,
                message=f"camera/video error: {error}\n",
            )
        except PoseModelLoadError as error:
            parser.exit(
                status=1,
                message=f"pose model error: {error}\n",
            )
        except SessionWriteError as error:
            parser.exit(
                status=1,
                message=f"session recording error: {error}\n",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
