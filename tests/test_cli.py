from argparse import ArgumentTypeError
from pathlib import Path

import pytest

import handstand_coach.cli as cli_module
from handstand_coach.capture import VideoSourceError
from handstand_coach.estimation import PoseModelLoadError
from handstand_coach.recording import SessionWriteError


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        ("1", 1),
        ("video.mp4", "video.mp4"),
        ("http://camera.local/stream", "http://camera.local/stream"),
    ],
)
def test_parse_source_distinguishes_camera_indices_and_paths(
    value: str,
    expected: int | str,
) -> None:
    assert cli_module.parse_source(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0.0),
        ("0.5", 0.5),
        ("1", 1.0),
    ],
)
def test_parse_confidence_threshold_accepts_normalized_values(
    value: str,
    expected: float,
) -> None:
    assert cli_module.parse_confidence_threshold(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "-0.1",
        "1.1",
        "not-a-number",
    ],
)
def test_parse_confidence_threshold_rejects_invalid_values(
    value: str,
) -> None:
    with pytest.raises(ArgumentTypeError):
        cli_module.parse_confidence_threshold(value)


def test_main_dispatches_live_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def fake_run_live(
        *,
        source: int | str | Path,
        model_path: str | Path,
        confidence_threshold: float,
        record_session: bool,
        output_directory: Path,
    ) -> None:
        received.update(
            source=source,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            record_session=record_session,
            output_directory=output_directory,
        )

    monkeypatch.setattr(cli_module, "run_live", fake_run_live)

    exit_code = cli_module.main(
        [
            "live",
            "--source",
            "1",
            "--model",
            "custom-pose.pt",
            "--confidence-threshold",
            "0.3",
            "--record-session",
            "--output-dir",
            "recorded-sessions",
        ]
    )

    assert exit_code == 0
    assert received == {
        "source": 1,
        "model_path": Path("custom-pose.pt"),
        "confidence_threshold": 0.3,
        "record_session": True,
        "output_directory": Path("recorded-sessions"),
    }


def test_main_reports_video_source_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_run_live(**_arguments: object) -> None:
        raise VideoSourceError("Unable to open video source: 99")

    monkeypatch.setattr(cli_module, "run_live", failing_run_live)

    with pytest.raises(SystemExit) as exit_result:
        cli_module.main(["live", "--source", "99"])

    captured = capsys.readouterr()

    assert exit_result.value.code == 1
    assert "camera/video error" in captured.err
    assert "Unable to open video source" in captured.err


def test_main_reports_pose_model_load_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_run_live(**_arguments: object) -> None:
        raise PoseModelLoadError("Unable to load pose model: missing-model.pt")

    monkeypatch.setattr(cli_module, "run_live", failing_run_live)

    with pytest.raises(SystemExit) as exit_result:
        cli_module.main(["live", "--model", "missing-model.pt"])

    captured = capsys.readouterr()

    assert exit_result.value.code == 1
    assert "pose model error" in captured.err
    assert "Unable to load pose model: missing-model.pt" in captured.err


def test_main_disables_recording_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def fake_run_live(**arguments: object) -> None:
        received.update(arguments)

    monkeypatch.setattr(cli_module, "run_live", fake_run_live)

    exit_code = cli_module.main(["live"])

    assert exit_code == 0
    assert received["record_session"] is False
    assert received["output_directory"] == Path("sessions")


def test_main_reports_session_write_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_run_live(**_arguments: object) -> None:
        raise SessionWriteError("Unable to initialize session: sessions/example")

    monkeypatch.setattr(cli_module, "run_live", failing_run_live)

    with pytest.raises(SystemExit) as exit_result:
        cli_module.main(["live", "--record-session"])

    captured = capsys.readouterr()

    assert exit_result.value.code == 1
    assert "session recording error" in captured.err
    assert "Unable to initialize session" in captured.err
