import argparse
from ast import parse
from pathlib import Path
from numpy import absolute
from ultralytics import YOLO


dataset_path = Path(__file__).parents[2].joinpath("data/out001/valid/images")
images = [file for file in dataset_path.glob("*")]
breakpoint()
model = YOLO("../../runs/segment/train2/weights/best.pt")
results = model(images[:15])
for i, result in enumerate(results):
    result.save(filename=f"frame_{i}.jpg")
