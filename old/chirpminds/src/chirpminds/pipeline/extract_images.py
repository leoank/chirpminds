"""Pipeline to extract images from video."""

from pathlib import Path
from subprocess import run as srun
from shlex import split


def extract_frames(in_video: Path, out_dir: Path) -> Path:
    """Extract frames from video file.

    Parameters
    ----------
    in_video : Path
        Path to read video file.
    output_path : Path
        Path to save output files.
    Returns
    -------
    Path
        Path to output file dir.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {in_video.resolve()} -r 2 -s 640x360 -q:v 2 {out_dir.resolve()}/frame%d.jpeg"
    cmd = split(cmd)
    res = srun(cmd)
    if res.returncode == 0:
        return out_dir
    else:
        raise Exception(res.stdout)


def process_video_files(vid_list: list[Path], out_dir: Path) -> None:
    """Process video files to extract frames.

    Parameters
    ----------
    vid_list : list[Path]
        List of video files to process.
    out_dir : Path
        Path to save output files.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for vid in vid_list:
        _ = extract_frames(vid, out_dir.joinpath(vid.stem))
