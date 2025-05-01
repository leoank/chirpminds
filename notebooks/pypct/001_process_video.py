# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Process raw video files

# %%
# | default_exp process_video

# %%
# | export
from pathlib import Path

import ffmpeg
import numpy as np
from tqdm import tqdm

from chirpminds.utils import parallel

# %% [markdown]
# ## Read video files

# %%
raw_data_dir = Path("../../scratch/raw_data")
frames_dir = Path("../../scratch/frames")
frames_dir.mkdir(exist_ok=True)
file_list_mts = [file for file in raw_data_dir.glob("*.MTS")]
file_list_mp4 = [file for file in raw_data_dir.glob("*.MP4")]

# %%
[file_list_mts[-1]] + file_list_mp4 


# %%
# | export
def get_video_info(video_path: Path) -> dict:
    probe = ffmpeg.probe(video_path.resolve().__str__())
    return next(s for s in probe["streams"] if s["codec_type"] == "video")


# %%
get_video_info(file_list[1])

# %% [markdown]
#  ## Process video files


# %%
# | export
def extract_frame(
    start_time_list: list[str], file_path: Path, out_dir: Path, job_idx: int = 0
) -> None:
    for start_time in start_time_list:
        ffmpeg.input(str(file_path.resolve()), ss=start_time).output(
            str(out_dir.resolve() / f"{file_path.stem}_{start_time}.jpg"), vframes=1
        ).run()


# %%
# | export


def extract_frames(
    video_path_list: list[Path], num_frames: int, out_dir: Path, job_idx: int = 0
) -> None:
    for video in tqdm(video_path_list, position=job_idx):
        print(f"Processing {video}")
        video_info = get_video_info(video)
        sampled_start_times = np.linspace(
            0, round(float(video_info["duration"])), num_frames, dtype=np.int32
        )
        parallel(sampled_start_times.tolist(), extract_frame, [video, out_dir])


# %%
extract_frames(file_list_mts[:-1], 160, frames_dir)


# %%
extract_frames([file_list_mts[-1]] + file_list_mp4 , 80, frames_dir)

# %% [markdown]
# ## View sampled frames

# %%
# | hide
import nbdev  # noqa

nbdev.nbdev_export()

# %%
