"""Read video frames."""

# %%
from chirpminds.data.io import yield_frames
from chirpminds.utils import get_default_data_folder
from torchvision import datapoints

# %%
video_file = get_default_data_folder().joinpath(
    "raw/pilot/feeder_videos/20230114_Feeder02A/MVI_0013.MP4"
)

# %%
for frame in yield_frames(video_file):
    frame_image = frame.to_image()
    break

# %%
frame_image
# %%
datapoints.Image(frame_image)
