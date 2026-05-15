from __future__ import annotations

import argparse
import bisect
import csv
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


REQUIRED_COLUMNS = {"frame_timestamp", "x_min", "y_min", "x_max", "y_max"}


@dataclass(frozen=True)
class BBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class TimestampBBoxes:
    timestamp_ms: float
    bboxes: list[BBox]


@dataclass(frozen=True)
class VideoMetadata:
    width: int | None = None
    height: int | None = None
    frame_times_ms: list[float] | None = None
    timing_source: str = "none"


def _parse_number(value: str) -> float:
    return float(value.strip().replace(",", "."))


def _parse_bbox(row: dict[str, str]) -> BBox:
    return BBox(
        x_min=round(_parse_number(row["x_min"])),
        y_min=round(_parse_number(row["y_min"])),
        x_max=round(_parse_number(row["x_max"])),
        y_max=round(_parse_number(row["y_max"])),
    )


def _clamp_bbox(bbox: BBox, width: int, height: int) -> BBox | None:
    x_min = max(0, min(width - 1, bbox.x_min))
    y_min = max(0, min(height - 1, bbox.y_min))
    x_max = max(0, min(width - 1, bbox.x_max))
    y_max = max(0, min(height - 1, bbox.y_max))

    if x_max <= x_min or y_max <= y_min:
        return None

    return BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _scale_bbox(
    bbox: BBox,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> BBox:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source bbox size must be positive")

    scale_x = target_width / source_width
    scale_y = target_height / source_height

    return BBox(
        x_min=round(bbox.x_min * scale_x),
        y_min=round(bbox.y_min * scale_y),
        x_max=round(bbox.x_max * scale_x),
        y_max=round(bbox.y_max * scale_y),
    )


def _draw_bboxes(frame, bboxes: Iterable[BBox]) -> None:
    for bbox in bboxes:
        cv2.rectangle(
            frame,
            (bbox.x_min, bbox.y_min),
            (bbox.x_max, bbox.y_max),
            color=(0, 255, 0),
            thickness=2,
        )


def _fourcc_candidates_for_output(output_path: Path) -> tuple[str, ...]:
    if output_path.suffix.lower() == ".avi":
        return ("MJPG", "XVID", "mp4v")
    if output_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        return ("mp4v", "avc1", "H264", "MJPG")
    return ("mp4v", "avc1", "H264", "MJPG")


def _open_video_writer(output_path: Path, fps: float, size: tuple[int, int]):
    tried_codecs = []

    for codec in _fourcc_candidates_for_output(output_path):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            size,
        )
        if writer.isOpened():
            return writer

        writer.release()
        tried_codecs.append(codec)

    raise RuntimeError(
        "Cannot open output video for writing: "
        f"{output_path}. Tried codecs: {', '.join(tried_codecs)}"
    )


def _iter_mp4_boxes(data: bytes, start: int, end: int):
    position = start

    while position + 8 <= end:
        size = struct.unpack(">I", data[position : position + 4])[0]
        box_type = data[position + 4 : position + 8].decode("latin1")
        header_size = 8

        if size == 1:
            if position + 16 > end:
                break
            size = struct.unpack(">Q", data[position + 8 : position + 16])[0]
            header_size = 16
        elif size == 0:
            size = end - position

        if size < header_size or position + size > end:
            break

        yield position, position + header_size, position + size, box_type
        position += size


def _parse_mp4_track(data: bytes, start: int, end: int) -> dict[str, object]:
    track: dict[str, object] = {}

    def walk(box_start: int, box_end: int) -> None:
        for _, content_start, content_end, box_type in _iter_mp4_boxes(
            data, box_start, box_end
        ):
            if box_type == "mdhd":
                version = data[content_start]
                offset = content_start + 4
                if version == 1:
                    offset += 8 + 8
                    timescale = struct.unpack(">I", data[offset : offset + 4])[0]
                else:
                    offset += 4 + 4
                    timescale = struct.unpack(">I", data[offset : offset + 4])[0]

                track["timescale"] = timescale
            elif box_type == "tkhd":
                version = data[content_start]
                offset = content_start + 4
                if version == 1:
                    offset += 8 + 8 + 4 + 4 + 8
                else:
                    offset += 4 + 4 + 4 + 4 + 4

                offset += 8 + 2 + 2 + 2 + 2 + 36
                track["width"] = round(
                    struct.unpack(">I", data[offset : offset + 4])[0] / 65536
                )
                track["height"] = round(
                    struct.unpack(">I", data[offset + 4 : offset + 8])[0] / 65536
                )
            elif box_type == "hdlr":
                track["handler_type"] = data[
                    content_start + 8 : content_start + 12
                ].decode("latin1")
            elif box_type == "stts":
                entry_count = struct.unpack(
                    ">I", data[content_start + 4 : content_start + 8]
                )[0]
                offset = content_start + 8
                entries = []

                for _ in range(entry_count):
                    sample_count, sample_delta = struct.unpack(
                        ">II", data[offset : offset + 8]
                    )
                    entries.append((sample_count, sample_delta))
                    offset += 8

                track["stts"] = entries
            elif box_type == "ctts":
                version = data[content_start]
                entry_count = struct.unpack(
                    ">I", data[content_start + 4 : content_start + 8]
                )[0]
                offset = content_start + 8
                entries = []

                for _ in range(entry_count):
                    sample_count = struct.unpack(">I", data[offset : offset + 4])[0]
                    if version == 1:
                        sample_offset = struct.unpack(
                            ">i", data[offset + 4 : offset + 8]
                        )[0]
                    else:
                        sample_offset = struct.unpack(
                            ">I", data[offset + 4 : offset + 8]
                        )[0]
                    entries.append((sample_count, sample_offset))
                    offset += 8

                track["ctts"] = entries
            elif box_type == "elst":
                version = data[content_start]
                entry_count = struct.unpack(
                    ">I", data[content_start + 4 : content_start + 8]
                )[0]
                offset = content_start + 8
                entries = []

                for _ in range(entry_count):
                    if version == 1:
                        segment_duration = struct.unpack(
                            ">Q", data[offset : offset + 8]
                        )[0]
                        media_time = struct.unpack(
                            ">q", data[offset + 8 : offset + 16]
                        )[0]
                        offset += 16
                    else:
                        segment_duration = struct.unpack(
                            ">I", data[offset : offset + 4]
                        )[0]
                        media_time = struct.unpack(
                            ">i", data[offset + 4 : offset + 8]
                        )[0]
                        offset += 8

                    media_rate_integer, media_rate_fraction = struct.unpack(
                        ">hH", data[offset : offset + 4]
                    )
                    offset += 4
                    entries.append(
                        (
                            segment_duration,
                            media_time,
                            media_rate_integer,
                            media_rate_fraction,
                        )
                    )

                track["elst"] = entries
            elif box_type in {"mdia", "minf", "stbl", "edts"}:
                walk(content_start, content_end)

    walk(start, end)
    return track


def _parse_mvhd_timescale(data: bytes, start: int, end: int) -> int | None:
    for _, content_start, _, box_type in _iter_mp4_boxes(data, start, end):
        if box_type != "mvhd":
            continue

        version = data[content_start]
        offset = content_start + 4
        if version == 1:
            offset += 8 + 8
        else:
            offset += 4 + 4

        return struct.unpack(">I", data[offset : offset + 4])[0]

    return None


def _expand_entries(entries: list[tuple[int, int]]) -> list[int]:
    values = []
    for sample_count, value in entries:
        values.extend([value] * sample_count)
    return values


def _build_mp4_presentation_times_ms(
    track: dict[str, object],
    movie_timescale: int | None,
) -> tuple[list[float] | None, str]:
    timescale = track.get("timescale")
    stts_entries = track.get("stts")
    if (
        not isinstance(timescale, int)
        or timescale <= 0
        or not isinstance(stts_entries, list)
        or not stts_entries
    ):
        return None, "none"

    dts_values = []
    current_time = 0
    for sample_count, sample_delta in stts_entries:
        for _ in range(sample_count):
            dts_values.append(current_time)
            current_time += sample_delta

    ctts_entries = track.get("ctts")
    if isinstance(ctts_entries, list) and ctts_entries:
        ctts_offsets = _expand_entries(ctts_entries)
        timing_source = "stts+ctts"
    else:
        ctts_offsets = [0] * len(dts_values)
        timing_source = "stts"

    sample_count = min(len(dts_values), len(ctts_offsets))
    presentation_values = [
        dts_values[index] + ctts_offsets[index] for index in range(sample_count)
    ]

    edit_entries = track.get("elst")
    presentation_pairs: list[tuple[float, int]] = []

    if (
        isinstance(edit_entries, list)
        and edit_entries
        and isinstance(movie_timescale, int)
        and movie_timescale > 0
    ):
        movie_cursor = 0
        for (
            segment_duration,
            media_time,
            media_rate_integer,
            media_rate_fraction,
        ) in edit_entries:
            if media_time < 0:
                movie_cursor += segment_duration
                continue
            if media_rate_integer == 0 and media_rate_fraction == 0:
                movie_cursor += segment_duration
                continue

            segment_duration_media = segment_duration * timescale / movie_timescale
            segment_end = media_time + segment_duration_media
            movie_cursor_ms = movie_cursor * 1000 / movie_timescale

            for sample_index, presentation_time in enumerate(presentation_values):
                if media_time <= presentation_time < segment_end:
                    time_ms = (
                        movie_cursor_ms
                        + (presentation_time - media_time) * 1000 / timescale
                    )
                    presentation_pairs.append((time_ms, sample_index))

            movie_cursor += segment_duration

        if presentation_pairs:
            timing_source += "+elst"

    if not presentation_pairs:
        if not presentation_values:
            return None, "none"

        first_presentation_time = min(presentation_values)
        presentation_pairs = [
            ((presentation_time - first_presentation_time) * 1000 / timescale, index)
            for index, presentation_time in enumerate(presentation_values)
        ]

    presentation_pairs.sort(key=lambda item: (item[0], item[1]))
    return [time_ms for time_ms, _ in presentation_pairs], timing_source


def _read_mp4_video_metadata(video_path: Path) -> VideoMetadata:
    if video_path.suffix.lower() not in {".mp4", ".m4v", ".mov"}:
        return VideoMetadata()

    data = video_path.read_bytes()
    tracks = []
    movie_timescale = None

    for _, content_start, content_end, box_type in _iter_mp4_boxes(data, 0, len(data)):
        if box_type != "moov":
            continue

        movie_timescale = _parse_mvhd_timescale(data, content_start, content_end)

        for _, trak_content_start, trak_content_end, trak_box_type in _iter_mp4_boxes(
            data, content_start, content_end
        ):
            if trak_box_type == "trak":
                tracks.append(_parse_mp4_track(data, trak_content_start, trak_content_end))

    for track in tracks:
        if track.get("handler_type") != "vide":
            continue

        width = track.get("width")
        height = track.get("height")
        timescale = track.get("timescale")
        entries = track.get("stts")
        if not isinstance(timescale, int) or not isinstance(entries, list) or not entries:
            return VideoMetadata(
                width=width if isinstance(width, int) and width > 0 else None,
                height=height if isinstance(height, int) and height > 0 else None,
            )

        frame_times, timing_source = _build_mp4_presentation_times_ms(
            track, movie_timescale
        )

        return VideoMetadata(
            width=width if isinstance(width, int) and width > 0 else None,
            height=height if isinstance(height, int) and height > 0 else None,
            frame_times_ms=frame_times,
            timing_source=timing_source,
        )

    return VideoMetadata()


def _choose_source_size(
    video_metadata: VideoMetadata,
    frame_width: int,
    frame_height: int,
    source_width: int | None,
    source_height: int | None,
) -> tuple[int, int]:
    if source_width is not None or source_height is not None:
        if source_width is None or source_height is None:
            raise ValueError("source_width and source_height must be set together")
        return source_width, source_height

    if video_metadata.width and video_metadata.height:
        return video_metadata.width, video_metadata.height

    return frame_width, frame_height


def _mp4_frame_index_at_or_after(
    timestamp_ms: float,
    frame_times_ms: list[float] | None,
) -> int | None:
    if frame_times_ms:
        position = bisect.bisect_left(frame_times_ms, timestamp_ms)
        if position < 0:
            return 0
        if position >= len(frame_times_ms):
            return len(frame_times_ms) - 1
        return position

    return None


def _cfr_frame_index_at_or_after(
    timestamp_ms: float,
    fps: float,
    frame_count: int,
) -> int:
    frame_index = math.ceil(timestamp_ms / 1000 * fps)
    if frame_count > 0:
        return max(0, min(frame_count - 1, frame_index))
    return max(0, frame_index)


def _timestamp_to_frame_index(
    timestamp_ms: float,
    fps: float,
    frame_count: int,
    frame_times_ms: list[float] | None,
    timestamp_mode: str,
) -> int:
    if timestamp_mode == "mp4":
        frame_index = _mp4_frame_index_at_or_after(timestamp_ms, frame_times_ms)
        if frame_index is not None:
            return frame_index

    return _cfr_frame_index_at_or_after(timestamp_ms, fps, frame_count)


def _load_bboxes_by_timestamp(
    csv_path: Path,
    frame_width: int,
    frame_height: int,
    source_width: int,
    source_height: int,
) -> list[TimestampBBoxes]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    bboxes_by_timestamp: dict[float, list[BBox]] = {}
    skipped = 0

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"CSV file is missing columns: {missing_columns}")

        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp_ms = _parse_number(row["frame_timestamp"])
                bbox = _scale_bbox(
                    _parse_bbox(row),
                    source_width=source_width,
                    source_height=source_height,
                    target_width=frame_width,
                    target_height=frame_height,
                )
            except (KeyError, TypeError, ValueError) as exc:
                skipped += 1
                print(
                    f"warning: skipped row {row_number}: cannot parse bbox ({exc})",
                    file=sys.stderr,
                )
                continue

            clamped_bbox = _clamp_bbox(bbox, frame_width, frame_height)
            if clamped_bbox is None:
                skipped += 1
                print(
                    f"warning: skipped row {row_number}: bbox is outside frame bounds",
                    file=sys.stderr,
                )
                continue

            bboxes_by_timestamp.setdefault(timestamp_ms, []).append(clamped_bbox)

    if skipped:
        print(f"warning: skipped {skipped} invalid CSV rows", file=sys.stderr)

    return [
        TimestampBBoxes(timestamp_ms=timestamp_ms, bboxes=bboxes)
        for timestamp_ms, bboxes in sorted(bboxes_by_timestamp.items())
    ]


def _bbox_bounds(items: list[TimestampBBoxes]) -> tuple[int, int, int, int] | None:
    bboxes = [bbox for item in items for bbox in item.bboxes]
    if not bboxes:
        return None

    return (
        min(bbox.x_min for bbox in bboxes),
        min(bbox.y_min for bbox in bboxes),
        max(bbox.x_max for bbox in bboxes),
        max(bbox.y_max for bbox in bboxes),
    )


def draw_bbox(
    video_path: str | Path,
    output_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    freeze_seconds: float = 1.0,
    source_width: int | None = None,
    source_height: int | None = None,
    timestamp_offset_ms: float = 0.0,
    timestamp_mode: str = "auto",
    freeze_mode: str = "auto",
    debug: bool = False,
) -> Path:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    csv_path = Path(csv_path) if csv_path is not None else video_path.with_suffix(".csv")
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path.cwd() / f"{video_path.stem}_bboxed.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if freeze_seconds < 0:
        raise ValueError("freeze_seconds must be non-negative")
    if timestamp_mode not in {"auto", "cfr", "mp4"}:
        raise ValueError("timestamp_mode must be 'auto', 'cfr', or 'mp4'")
    if freeze_mode not in {"auto", "replace", "insert"}:
        raise ValueError("freeze_mode must be 'auto', 'replace', or 'insert'")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            raise RuntimeError(f"Cannot determine FPS for video: {video_path}")
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Cannot determine frame size for video: {video_path}")

        video_metadata = _read_mp4_video_metadata(video_path)
        source_width, source_height = _choose_source_size(
            video_metadata=video_metadata,
            frame_width=width,
            frame_height=height,
            source_width=source_width,
            source_height=source_height,
        )
        effective_timestamp_mode = (
            "mp4"
            if timestamp_mode == "auto" and video_metadata.frame_times_ms
            else "cfr"
            if timestamp_mode == "auto"
            else timestamp_mode
        )
        effective_freeze_mode = "insert" if freeze_mode == "auto" else freeze_mode

        if debug:
            print(
                "debug: "
                f"frame_size={width}x{height}, fps={fps:.6g}, "
                f"frame_count={frame_count}, "
                f"mp4_size={video_metadata.width}x{video_metadata.height}, "
                f"mp4_timing={video_metadata.timing_source}, "
                f"source_size={source_width}x{source_height}, "
                f"timestamp_mode={effective_timestamp_mode}, "
                f"freeze_mode={effective_freeze_mode}, "
                f"timestamp_offset_ms={timestamp_offset_ms}",
                file=sys.stderr,
            )

        bboxes_by_timestamp = _load_bboxes_by_timestamp(
            csv_path=csv_path,
            frame_width=width,
            frame_height=height,
            source_width=source_width,
            source_height=source_height,
        )
        if debug:
            bounds = _bbox_bounds(bboxes_by_timestamp)
            print(
                "debug: "
                f"timestamps={len(bboxes_by_timestamp)}, "
                f"bbox_bounds={bounds}, "
                f"scale_x={width / source_width:.6g}, "
                f"scale_y={height / source_height:.6g}",
                file=sys.stderr,
            )

        frame_times_ms = video_metadata.frame_times_ms
        bboxes_by_frame: dict[int, list[BBox]] = {}
        if frame_times_ms and bboxes_by_timestamp:
            min_timestamp = bboxes_by_timestamp[0].timestamp_ms + timestamp_offset_ms
            max_timestamp = bboxes_by_timestamp[-1].timestamp_ms + timestamp_offset_ms
            min_frame_time = frame_times_ms[0]
            max_frame_time = frame_times_ms[-1]
            if min_timestamp < min_frame_time or max_timestamp > max_frame_time:
                print(
                    "warning: CSV timestamps are outside video presentation time "
                    f"range: csv=[{min_timestamp:g}, {max_timestamp:g}] ms, "
                    f"video=[{min_frame_time:g}, {max_frame_time:g}] ms",
                    file=sys.stderr,
                )

        for timestamp_bboxes in bboxes_by_timestamp:
            frame_index = _timestamp_to_frame_index(
                timestamp_bboxes.timestamp_ms + timestamp_offset_ms,
                fps=fps,
                frame_count=frame_count,
                frame_times_ms=frame_times_ms,
                timestamp_mode=effective_timestamp_mode,
            )
            bboxes_by_frame.setdefault(frame_index, []).extend(timestamp_bboxes.bboxes)

            if debug:
                print(
                    "debug: "
                    f"timestamp_ms={timestamp_bboxes.timestamp_ms:g} -> "
                    f"frame={frame_index}, boxes={len(timestamp_bboxes.bboxes)}",
                    file=sys.stderr,
                )

        writer = _open_video_writer(output_path, fps, (width, height))

        try:
            freeze_frames = max(1, round(freeze_seconds * fps))
            frame_index = 0

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index in bboxes_by_frame:
                    annotated_frame = frame.copy()
                    _draw_bboxes(annotated_frame, bboxes_by_frame[frame_index])
                    for _ in range(freeze_frames):
                        writer.write(annotated_frame)

                    if effective_freeze_mode == "replace":
                        for _ in range(freeze_frames - 1):
                            ok, _ = capture.read()
                            if not ok:
                                break
                            frame_index += 1
                else:
                    writer.write(frame)

                frame_index += 1
        finally:
            writer.release()
    finally:
        capture.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            "Output video was not written. Current OpenCV build cannot encode "
            f"this output format: {output_path}"
        )

    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes from a sidecar CSV onto an mp4 video."
    )
    parser.add_argument("video_path", help="Path to the input .mp4 file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output video path. Defaults to ./<original_filename>_bboxed.mp4.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="CSV path override. Defaults to a CSV next to the video with the same name.",
    )
    parser.add_argument(
        "--freeze-seconds",
        type=float,
        default=1.0,
        help="How many seconds to freeze each annotated frame. Defaults to 1.0.",
    )
    parser.add_argument(
        "--source-width",
        type=int,
        default=None,
        help="BBox coordinate source width. Defaults to MP4 metadata width.",
    )
    parser.add_argument(
        "--source-height",
        type=int,
        default=None,
        help="BBox coordinate source height. Defaults to MP4 metadata height.",
    )
    parser.add_argument(
        "--timestamp-offset-ms",
        type=float,
        default=0.0,
        help=(
            "Shift CSV timestamps before matching frames. Positive values draw "
            "later, negative values draw earlier. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=("auto", "cfr", "mp4"),
        default="auto",
        help=(
            "How to map CSV timestamps to frame indexes. 'auto' uses exact "
            "MP4 presentation timing when available and falls back to CFR. "
            "'mp4' forces MP4 timing; 'cfr' forces constant-FPS timing."
        ),
    )
    parser.add_argument(
        "--freeze-mode",
        choices=("auto", "replace", "insert"),
        default="auto",
        help=(
            "How freeze frames affect the timeline. 'replace' keeps later "
            "timestamps aligned by skipping the covered source frames. 'insert' "
            "pauses and lengthens the output. 'auto' uses insert."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detected video size, source size, and timestamp-frame mapping.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        output_path = draw_bbox(
            video_path=args.video_path,
            output_path=args.output,
            csv_path=args.csv_path,
            freeze_seconds=args.freeze_seconds,
            source_width=args.source_width,
            source_height=args.source_height,
            timestamp_offset_ms=args.timestamp_offset_ms,
            timestamp_mode=args.timestamp_mode,
            freeze_mode=args.freeze_mode,
            debug=args.debug,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
