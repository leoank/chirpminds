"""Automatic annotation with large models."""
# pyright: reportMissingTypeStubs=false

from pathlib import Path

from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill_grounded_sam_2 import GroundedSAM2
from supervision import DetectionDataset


def gen_annotations_sam(
    ontology: CaptionOntology, img_dir: Path, out_dir: Path
) -> DetectionDataset:
    """Generate annoations with GroundedSAM.

    Parameters
    ----------
    ontology : CaptionOntology
        Ontology to use for generating annotations.
    img_dir : Path
        Path to images to annotate.
    out_dir : Path
        Path to save output files.

    Returns
    -------
    DetectionDataset
        Generated DetectionDataset.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model = GroundedSAM(ontology)
    dataset = model.label(
        str(img_dir.resolve()), extension=".jpeg", output_folder=str(out_dir.resolve())
    )
    return dataset


def gen_annotations_sam2(
    ontology: CaptionOntology, img_dir: Path, out_dir: Path
) -> DetectionDataset:
    """Generate annoations with GroundedSAM2.

    Parameters
    ----------
    ontology : CaptionOntology
        Ontology to use for generating annotations.
    img_dir : Path
        Path to images to annotate.
    out_dir : Path
        Path to save output files.

    Returns
    -------
    DetectionDataset
        Generated DetectionDataset.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model = GroundedSAM2(ontology)
    dataset = model.label(
        str(img_dir.resolve()), extension=".jpeg", output_folder=str(out_dir.resolve())
    )
    return dataset
