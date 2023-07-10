"""Data IO."""

from collections.abc import Iterator
from pathlib import Path

import av


def yield_frames(file_path: Path) -> Iterator[av.VideoFrame]:
    """Generate frames from a video file.

    Parameters
    ----------
    file_path : Path
        Path to video file.wa

    Yields
    ------
    Iterator[av.VideoFrame]
        Video frame.
    """
    video_container = av.open(str(file_path.absolute()))
    video_container.streams.video[0].thread_type = "AUTO"
    yield from video_container.decode(video=0)
    video_container.close()
