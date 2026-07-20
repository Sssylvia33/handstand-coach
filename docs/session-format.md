# Session data format

## Purpose

The session format stores structured pose-estimation results from a live session so they can be analyzed later without rerunning the camera or pose model.

The format preserves frame order, timestamps, image dimensions, detected keypoints, confidence values, and frames where no pose was detected.

## Directory structure

Each recording has its own session directory:

```text
sessions/
`-- 2026-07-20T183000Z/
    |-- metadata.json
    `-- poses.jsonl
```

`metadata.json` contains information that applies to the entire session.

`poses.jsonl` contains one pose-frame record per line.

## Session metadata

Example `metadata.json`:

```json
{
  "schema_version": 1,
  "session_id": "2026-07-20T183000Z",
  "started_at_utc": "2026-07-20T18:30:00Z",
  "source": "0",
  "model": "yolov8n-pose.pt",
  "confidence_threshold": 0.5
}
```

## Frame record

Each line in `poses.jsonl` is an independent JSON object.

Frame without a detected pose:

```json
{"frame_index":0,"timestamp_s":0.0,"image_width":1280,"image_height":720,"pose":null}
```

Frame with a detected pose:

```json
{"frame_index":1,"timestamp_s":0.08,"image_width":1280,"image_height":720,"pose":{"keypoints":[{"name":"left_shoulder","x":0.42,"y":0.31,"confidence":0.96}]}}
```

The keypoint coordinates are normalized to the range from `0.0` to `1.0`.

## Missing pose behavior

A frame without a detected person is retained with:

```json
"pose": null
```

JSON `null` is converted to Python `None` when the record is loaded.

The frame is not discarded because its index and timestamp are part of the session timeline. This allows later analysis to distinguish a missing pose from a missing frame.

## Format decision

Session metadata is stored as ordinary JSON because it is one small structured object that is read as a unit.

Pose frames are stored as JSON Lines because frames are produced incrementally. Each frame can be appended immediately without keeping the complete session in memory.

JSON Lines was selected because it is:

- Append-friendly
- Human-readable
- Easy to process one record at a time
- Partially recoverable if recording is interrupted
- Supported by Python's standard `json` module

Alternatives considered:

- A single JSON array would normally require finalizing the complete document after recording.
- CSV does not represent nested poses and keypoints naturally.
- A database or Parquet file would add complexity that is unnecessary for the current project scale.

The main disadvantage of JSON Lines is larger files and slower analytical queries compared with binary formats. This tradeoff is acceptable for the current MVP.

## Schema evolution

Session files may remain after the application code changes. The `schema_version` tells a reader which data structure and validation rules apply.

Version 1 defines:

- One analyzed person at most
- Normalized two-dimensional keypoints
- Confidence values for individual keypoints
- An explicit `null` value when no pose is detected

Future incompatible format changes must increase `schema_version`. For example, supporting multiple people might require schema version 2.

Readers must reject unsupported schema versions with a clear error instead of silently interpreting the data incorrectly. The meaning of an existing schema version must not be changed after session files have been created.