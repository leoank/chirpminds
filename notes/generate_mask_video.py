"""Generate mask video."""
# %%
from typing import Any

import av
import numpy as np
import rerun as rr
from chirpminds.data.io import yield_frames
from chirpminds.pipeline.gen_mask import gen_mask
from chirpminds.utils import get_default_data_folder
from tqdm import tqdm

rr.init("chirpminds", spawn=True)


# %%
video_file = get_default_data_folder().joinpath(
    "raw/pilot/feeder_videos/20230114_Feeder02A/MVI_0013.MP4"
)

read_container = av.open(str(video_file))
read_container_stream = read_container.streams[0]

# %%
write_container = av.open(
    str(
        get_default_data_folder().joinpath(
            "raw/pilot/feeder_videos/20230114_Feeder02A/MVI_0013_mask.MP4"
        )
    ),
    mode="w",
)

write_stream = write_container.add_stream(
    "mpeg4", rate=read_container_stream.average_rate
)
# write_stream = write_container.add_stream(template=read_container_stream)
write_stream.width = read_container_stream.width
write_stream.height = read_container_stream.height
write_stream.pix_fmt = read_container_stream.pix_fmt


# %%
def write_frame(frame: np.ndarray, container: Any, stream: Any) -> None:  # noqa: ANN401
    """Write frame to container stream.

    Parameters
    ----------
    frame : np.ndarray
        Frame as a numpy array.
    container : Any
        PyAV Video container.
    stream : Any
        PyAV Video stream.
    """
    frame = np.round(255 * frame).astype(np.uint8)
    frame = np.clip(frame, 0, 255)
    frame = av.VideoFrame.from_ndarray(frame, format="gray8")
    for packet in stream.encode(frame):
        container.mux(packet)


def write_frames(
    frames: np.ndarray, container: Any, stream: Any  # noqa: ANN401
) -> None:
    """Write frames to container stream.

    Parameters
    ----------
    frames : np.ndarray
        Frame as a numpy array.
    container : Any
        PyAV Video container.
    stream : Any
        PyAV Video stream.
    """
    for frame in frames:
        write_frame(frame, container, stream)


# %%
count = 0
for frame in tqdm(yield_frames(video_file), total=300):
    frame = frame.to_image()
    rr.log_image("raw/video", frame)
    masks, boxes = gen_mask(frame)
    masks = masks.sum(0)
    framer = np.round(255 * masks[0]).astype(np.uint8)
    framer = np.clip(framer, 0, 255)
    rr.log_image("mask/video", framer)
    write_frames(masks, write_container, write_stream)
    if count == 300:
        break
    count += 1

# %%
for packet in write_stream.encode():
    write_container.mux(packet)
# %%
write_container.close()
# %%
