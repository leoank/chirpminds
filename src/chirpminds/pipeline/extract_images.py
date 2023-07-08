"""Pipeline to extract images from video."""

from pathlib import Path


def extract_images(
    input_path: Path, output_path: Path, force: bool, debug: bool
) -> Path:
    """Extract images from video files.

    Parameters
    ----------
    input_path : Path
        Path to read raw data files.
    output_path : Path
        Path to save output files.
    force : bool
        Force re run the pipeline and overwrite existing data.
    debug : bool
        Run in debug mode.

    Returns
    -------
    Path
        Path to output file dir.
    """
    pass
