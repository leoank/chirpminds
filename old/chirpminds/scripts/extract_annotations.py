"""Automatically extract refined annotations from a label studio project."""

import argparse
from pathlib import Path

import numpy as np
import supervision as sv
import torch as t
from label_studio_sdk.client import LabelStudio
from label_studio_sdk.converter.brush import (
    decode_rle,
)
from tqdm import tqdm

labels_dict = {"cbird_on_antenna": 0}


def extract_annotations(
    dataset: sv.DetectionDataset, api_key: str | None = None
) -> sv.DetectionDataset:
    # if api_key is None:
    #     API_KEY = os.environ.get("LABEL_STUDIO_KEY")
    # else:
    #     API_KEY = api_key

    API_KEY = "14acfa7569d9a020f4f8a0fe0cbaa25d72261b0f"

    client = LabelStudio(api_key=API_KEY, base_url="http://localhost:8080")

    # Create project
    project = client.projects.get(id=10)

    # Get task
    tasks = client.tasks.list(project=project.id)
    for i, task in enumerate(tqdm(tasks)):
        try:
            detections = dataset.annotations[
                input_data_path.joinpath(
                    "images/val", task.file_upload.split("-")[1]
                ).__str__()
            ]
        except KeyError:
            continue
        if len(task.annotations) == 0:
            detections = None
        else:
            print("generating masks")
            dets = []
            for anno in task.annotations:
                for res in anno["result"]:
                    mask = decode_rle(res["value"]["rle"])
                    mask = np.reshape(mask, [1080, 1920, 4])[:, :, 3]
                    mask = mask / 255
                    mask = mask.astype(bool)
                    mask = t.tensor(np.array([mask]))
                    labels = t.tensor(np.array([0]))
                    scores = t.tensor(np.array([0.99987]))
                    det = sv.Detections.from_transformers(
                        {"scores": scores, "labels": labels, "masks": mask},
                        {0: "cbird_on_antenna"},
                    )
                    dets.append(det)
            detections = sv.Detections.merge(dets)
        dataset.annotations[
            input_data_path.joinpath(
                "images/val", task.file_upload.split("-")[1]
            ).__str__()
        ] = detections
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup studio CLI.")
    parser.add_argument("in_folder", type=str, help="Path to input data folder")
    args = parser.parse_args()
    input_data_path = Path(args.in_folder).resolve().absolute()
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=str(input_data_path.joinpath("images/val")),
        annotations_directory_path=str(input_data_path.joinpath("labels/val")),
        data_yaml_path=str(input_data_path.joinpath("dataset.yaml")),
    )
    dataset = extract_annotations(dataset)

    out_path = input_data_path.parent.joinpath("cbirdsegval")
    dataset.as_yolo(
        images_directory_path=str(out_path.joinpath("images/val")),
        annotations_directory_path=str(out_path.joinpath("labels/val")),
        data_yaml_path=str(out_path.joinpath("dataset.yaml")),
    )
