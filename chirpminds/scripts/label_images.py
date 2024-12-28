import argparse
from ast import parse
from pathlib import Path
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from numpy import absolute

base_model = GroundedSAM(
    ontology=CaptionOntology(
        {
            "chickadee": "chickadee"
        }
    ),
    device="cuda"
)

def create_dataset(input_path: Path, output_path: Path):
    dataset = base_model.label(str(input_path.absolute()), extension=".jpg", output_folder=str(output_path.absolute()))
    dataset.as_yolo()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A CLI tool to automatic label images")
    parser.add_argument("image_folder", type=str, help="Path to input image folder" )
    parser.add_argument("output_folder", type=str, help="Path to output folder" )
    args = parser.parse_args()
    create_dataset(Path(args.image_folder), Path(args.output_folder))
