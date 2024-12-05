"""Chirpminds cli."""
# pyright: basic

import json
from pathlib import Path

import click

from chirpminds.pipeline.auto_annotation import CaptionOntology, gen_annotations_sam2
from chirpminds.pipeline.extract_images import extract_frames


@click.command()
@click.option("--inp", type=str, required=True)
@click.option("--out", type=str, required=True)
@click.option(
    "--prompt", type=str, default='{"A bird that resembles chickadee": "chickadee"}'
)
def annotate(inp: str, out: str, prompt: str) -> None:  # noqa: D103
    ontology = json.loads(prompt)
    _ = gen_annotations_sam2(CaptionOntology(ontology), Path(inp), Path(out))
    print(f"Find generated annotations at: {Path(out).resolve()}")


@click.command()
@click.option("--inp", type=str, required=True)
@click.option("--out", type=str, required=True)
def prepare(inp: str, out: str) -> None:  # noqa: D103
    out_dir = extract_frames(Path(inp), Path(out))
    print(f"Find generated frames at: {out_dir.resolve()}")


@click.group()
def main() -> None:
    """Chirpminds console."""
    pass


main.add_command(prepare)
main.add_command(annotate)
