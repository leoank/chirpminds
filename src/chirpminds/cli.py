"""Chirpminds cli."""
from pathlib import Path

import click

from chirpminds.pipeline.extract_images import extract_images

pipeline_modules_map = {
    "extract_images": extract_images,
}


def str2array(ctx: click.Context, _: str, value: str) -> list:
    """Convert string input by user into an array of values.

    Parameters
    ----------
    ctx : click.Context
        Click execution context.
    value : str
        Value received for the parameter.

    Returns
    -------
    list
        list of values.

    Raises
    ------
    click.BadParameter
        Raises an error for malformed input.
    """
    try:
        data_modules = [k for k in pipeline_modules_map.keys()]
        if value == "":
            config_list = data_modules
        elif value.startswith("~"):
            value = value.lstrip("~")
            config_list = value.split(",")
            for val in config_list:
                if val not in data_modules:
                    raise ValueError
            config_list = list(set(data_modules) - set(config_list))
        else:
            config_list = value.split(",")
            for val in config_list:
                if val not in data_modules:
                    raise ValueError
        return config_list
    except ValueError:
        raise click.BadParameter(f"{val} is an unknown data module")


@click.command()
@click.option(
    "-i",
    "--in_",
    type=click.Path(),
    help="Path to read input files",
    required=True,
)
@click.option(
    "-o",
    "--out",
    type=click.Path(),
    help="Path to save output files",
    required=True,
)
@click.option(
    "-c",
    "--config",
    type=click.UNPROCESSED,
    callback=str2array,
    help="""Comma separated list of pipeline modules to select/de-select.
            Example: extract_images or ~extract_images""",
    default="",
    show_default=True,
)
@click.option("-f", "--force", is_flag=True, help="Force redownload all data")
@click.option("-d", "--debug", is_flag=True, help="Run in debug mode.")
def process(in_: str, out: str, config: list, force: bool, debug: bool) -> None:
    """Process datasets."""
    for pipeline in config:
        pipeline_modules_map[pipeline](Path(in_), Path(out), force, debug)


@click.group()
def data() -> None:
    """Chirpminds data toolkit."""
    pass


@click.group()
def main() -> None:
    """Chirpminds console."""
    pass


data.add_command(process)
main.add_command(data)
