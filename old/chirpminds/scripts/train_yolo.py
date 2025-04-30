import argparse
from ast import parse
from pathlib import Path
from numpy import absolute

from ultralytics import YOLO

model = YOLO("yolo11x-seg.pt")

def train_model(input_path: Path):
    results = model.train(data=input_path.joinpath("data.yaml").absolute().__str__(), epochs=100, imgsz=[1080,1920], device=[0,1,2,3])
    breakpoint()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A CLI tool to finetune YOLO")
    parser.add_argument("data_folder", type=str, help="Path to data folder" )
    args = parser.parse_args()
    train_model(Path(args.data_folder))
