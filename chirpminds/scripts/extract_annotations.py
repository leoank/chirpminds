"""Automatically extract refined annotations from a label studio project."""

import argparse
from pathlib import Path
import os

import torch as t
import numpy as np
from label_studio_sdk.client import LabelStudio
from label_studio_sdk.converter.brush import (
    mask2rle,
    decode_from_annotation,
    decode_rle,
)
import supervision as sv
from tqdm import tqdm


def extract_annotations(
    dataset: sv.DetectionDataset, api_key: str | None = None
) -> sv.DetectionDataset:
    # if api_key is None:
    #     API_KEY = os.environ.get("LABEL_STUDIO_KEY")
    # else:
    #     API_KEY = api_key

    API_KEY = "cb6fd0c37537b04e945e969babc98b50b2fb1386"

    client = LabelStudio(api_key=API_KEY)

    # Create project
    project = client.projects.get(id=46)

    # Get task
    tasks = client.tasks.list(project=project.id)
    for i, task in enumerate(tqdm(tasks)):
        if i == 104:
            break
        detections = dataset.annotations[task.storage_filename]
        # breakpoint()
        if len(task.annotations) == 0:
            detections = None
        else:
            print("generating masks")
            dets = []
            for res in task.annotations[0]["result"]:
                mask = decode_rle(res["value"]["rle"])
                mask = np.reshape(mask, [1080, 1920, 4])[:, :, 3]
                mask = mask / 255
                mask = mask.astype(bool)
                mask = t.tensor(np.array([mask]))
                labels = t.tensor(np.array([0]))
                scores = t.tensor(np.array([0.99987]))
                det = sv.Detections.from_transformers(
                    {"scores": scores, "labels": labels, "masks": mask},
                    {0: "chickadee"},
                )
                dets.append(det)
            detections = sv.Detections.merge(dets)
        dataset.annotations[task.storage_filename] = detections
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup studio CLI.")
    parser.add_argument("in_folder", type=str, help="Path to input data folder")
    args = parser.parse_args()
    input_data_path = Path(args.in_folder).resolve().absolute()
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=str(input_data_path.joinpath("train/images")),
        annotations_directory_path=str(input_data_path.joinpath("train/labels")),
        data_yaml_path=str(input_data_path.joinpath("data.yaml")),
    )
    dataset = extract_annotations(dataset)

    out_path = input_data_path.parent.joinpath("ref001")
    dataset.as_yolo(
        images_directory_path=str(out_path.joinpath("train/images")),
        annotations_directory_path=str(out_path.joinpath("train/labels")),
        data_yaml_path=str(out_path.joinpath("data.yaml")),
    )
