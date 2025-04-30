"""Train models with autodistill."""
# pyright: reportMissingTypeStubs=false

from pathlib import Path
from autodistill_yolonas import YOLONAS
from autodistill_yolov8 import YOLOv8


def train_yolov8(
    initial_wt: Path, dataset_dir: Path, epochs: int, device: str = "cpu"
) -> None:
    target_model = YOLOv8(str(initial_wt.resolve()))
    target_model.train(str(dataset_dir.resolve()), epochs=epochs, device=device)  # pyright: ignore[reportUnknownMemberType]


def train_yolonas(
    initial_wt: Path, dataset_dir: Path, epochs: int, device: str = "cpu"
) -> None:
    target_model = YOLONAS(str(initial_wt.resolve()))
    target_model.train(str(dataset_dir.resolve()), epochs=epochs)
