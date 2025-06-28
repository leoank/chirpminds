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
from moviepy import VideoFileClip, concatenate_videoclips
from tqdm import tqdm
import json

from chirpminds.utils import parallel

# %% [markdown]
# ## Read video files

# %%
raw_data_dir = Path("../../scratch/raw_data")
clips_dir = Path("../../scratch/clips")
clips_dir.mkdir(exist_ok=True)
frames_dir = Path("../../scratch/frames")
frames_dir.mkdir(exist_ok=True)
frames_dir = Path("../../scratch/new_frames")
frames_dir.mkdir(exist_ok=True)
interlaced_frames_dir = Path("../../scratch/interlaced_frames_bwdif")
interlaced_frames_dir.mkdir(exist_ok=True)
file_list_mts = [
    file for file in raw_data_dir.glob("*.MTS") if file.name != "interlaced.MTS"
]
file_list_mp4 = [
    file for file in raw_data_dir.glob("*.MP4") if file.name != "interlaced.MTS"
]
interlaced_list_mts = [file for file in raw_data_dir.glob("interlaced.MTS")]

# %%
file_list_mp4

# %%
file_list_mts

# %% [markdown]
# ## View video

# %%
video_clip_1 = VideoFileClip(file_list_mts[0])
video_clip_2 = VideoFileClip(file_list_mts[1])
video_clip_3 = VideoFileClip(file_list_mts[2])

# %%
video_clip_3.subclipped(600).display_in_notebook(
    width=500, maxduration=300, fps=10, rd_kwargs=dict(bitrate="50k")
)

# %% [markdown]
# ## Make video clips

# %%
video_clip_1_clipped = video_clip_1.subclipped(800, 960)
video_clip_1_clipped.write_videofile(
    str(clips_dir.joinpath(file_list_mts[0].resolve().with_suffix(".mp4").name)),
    audio=False,
    write_logfile=True,
)
video_clip_1_clipped.close()

# %%
video_clip_2_clipped = video_clip_2.subclipped(0, 200)
video_clip_2_clipped.write_videofile(
    str(clips_dir.joinpath(file_list_mts[1].resolve().with_suffix(".mp4").name)),
    audio=False,
    write_logfile=True,
)
video_clip_2_clipped.close()

# %%
video_clip_3_concat = video_clip_3.subclipped(120, 190)
video_clip_3_concat.write_videofile(
    str(clips_dir.joinpath(file_list_mts[2].resolve().with_suffix(".mp4").name)),
    audio=False,
    write_logfile=True,
)
video_clip_3_concat.close()

# %% [markdown]
#  ## Process video files


# %%
# | export
def get_video_info(video_path: Path) -> dict:
    probe = ffmpeg.probe(video_path.resolve().__str__())
    return next(s for s in probe["streams"] if s["codec_type"] == "video")


# %%
# | export
def extract_frame(
    start_time_list: list[str],
    file_path: Path,
    out_dir: Path,
    quiet: bool = False,
    vf: str | None = None,
    job_idx: int = 0,
) -> None:
    for start_time in start_time_list:
        ffmpeg.input(str(file_path.resolve()), ss=start_time).output(
            str(out_dir.resolve() / f"{file_path.stem}_{start_time}.jpg"),
            vframes=1,
        ).run(overwrite_output=True, quiet=quiet)


# %%
# | export
def extract_frames(
    video_path_list: list[Path],
    num_frames: int,
    out_dir: Path,
    quiet: bool = False,
    vf: str | None = None,
    load_frames_path: Path | None = None,
    jobs: int = 2,
    job_idx: int = 0,
) -> None:
    for video in tqdm(video_path_list, position=job_idx):
        print(f"Processing {video}")
        video_info = get_video_info(video)
        if load_frames_path:
            frame_dict = json.load(load_frames_path.open())
            sampled_start_times = frame_dict[video.name.split(".")[0]]
        else:
            sampled_start_times = np.linspace(
                0, round(float(video_info["duration"])), num_frames + 1, dtype=np.int32
            )
        sampled_start_times = list(set(sampled_start_times))
        sampled_start_times.sort()
        parallel(
            sampled_start_times,
            extract_frame,
            [video, out_dir, quiet, vf],
            jobs,
        )


# %% [markdown]
# ## Process clips

# %%
clips = [file for file in clips_dir.glob("*.mp4")]
clips

# %%
get_video_info(clips[2])

# %%
# Change the number of jobs to 2.
# More number of threads tries to write to the same file and crash
extract_frames(clips, 133, frames_dir, True, 10)


# %% [markdown]
# ## Process full videos

# %%
extract_frames(file_list_mp4, 80, frames_dir, True, load_frames_path=Path("video_frames.json"), jobs=1)

# %%
extract_frames(file_list_mts, 133, frames_dir, True, load_frames_path=Path("video_frames.json"), jobs=1)

# %%
get_video_info(interlaced_list_mts[0])

# %% [markdown]
# ## View sampled frames

# %%
# | hide
import nbdev  # noqa

nbdev.nbdev_export()
