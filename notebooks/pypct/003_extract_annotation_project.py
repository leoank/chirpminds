# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Setup labelstudio project for annotation

# %%
# | default_exp labelstudio/annotation

# %%
# | export

from pathlib import Path
from shutil import copy

import cv2
import numpy as np
import supervision as sv
import torch as t
import yaml
from label_studio_sdk.client import LabelStudio
from label_studio_sdk.converter.brush import decode_rle
from label_studio_sdk.types import Task
from supervision.dataset.formats.yolo import detections_to_yolo_annotations
from tqdm import tqdm

from chirpminds.utils import parallel

# %% [markdown]
# # Labelstudio annotation extraction


# %% [markdown]
# ## Setting up env vars

# %%
api_key = "b0429a22dacfad6c4fe937b8d9ddcf7b8dce7de7"
base_url = "http://karkinos:8080"
model_url = "http://10.13.84.1:9090"
labels = ["bcch", "antenna", "bcch_on_aantenna", "bcch_off_antenna"]
# Path inside the container
frames_path = Path("/datastore/frames")
client = LabelStudio(api_key=api_key, base_url=base_url)
project_id = 10
local_frames_path = Path(
    "/home/ank/workspace/hub/leoank/chirpminds/main/scratch/new_frames"
)
detection_dataset_path = Path(
    "/home/ank/workspace/hub/leoank/chirpminds/main/scratch/cbird_anno_001"
)


# %% [markdown]
# ## Get annotations back from labelstudio

# %% [markdown]
# ```json
# "id": 902384,
# "storage_filename": "/datastore/frames/somefile.jpg",
# "predictions": [],
# "annotations": [
#     {
#         "id": 984,
#         "result": [
#             {
#                 "original_width": 1920,
#                 "original_height": 1080,
#                 "image_rotation": 0,
#                 "value": {
#                     "format": "rle",
#                     "rle": [...],
#                     "brushlabels": [ "bcch_off_antenna"]
#                 },
#                 "id": "09wer9092",
#                 "from_name": "label",
#                 "to_name": "img",
#                 "type": "brushlabels",
#                 "origin": "manual",
#                 "score": 0.876836
#             }
#         ],
#     }
# ],
# ```


# %%
# | export
def split_array(
    arr: list, train_ratio: float = 0.7, val_ratio: float = 0.2
) -> tuple[list, list, list]:
    """Split array into train/validation/test sets."""
    n = len(arr)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return arr[:train_end], arr[train_end:val_end], arr[val_end:]


# %%
# | export
def create_detection_yolo(labels: list[str], frames_path: Path, out_path: Path) -> None:
    # Create directories
    out_path.joinpath("annotated").mkdir(parents=True, exist_ok=True)
    for dir in ["train", "val", "test"]:
        detection_frames_path = out_path / f"{dir}/images"
        detection_labels_path = out_path / f"{dir}/labels"
        detection_frames_path.mkdir(parents=True, exist_ok=True)
        detection_labels_path.mkdir(parents=True, exist_ok=True)

    # Copy existing frames
    all_frames = [file for file in frames_path.glob("*.jpg")]
    train, val, test = split_array(all_frames)
    for split, dir_path in [
        (train, out_path.joinpath("train/images")),
        (val, out_path.joinpath("val/images")),
        (test, out_path.joinpath("test/images")),
    ]:
        for frame in split:
            copy(frame, dir_path.joinpath(frame.name))

    # Write the yaml file
    data_dict = {
        "path": out_path.absolute().__str__(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": [{i: label} for i, label in enumerate(labels)],
    }
    yaml_out_path = out_path / "data.yaml"
    yaml_out_path.write_text(yaml.dump(data_dict))


# %%
# | export
def write_annotated_image(
    detections: sv.Detections, image_path: Path, out_path: Path
) -> None:
    image = cv2.imread(image_path.absolute().__str__())
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_frame = box_annotator.annotate(scene=image, detections=detections)
    annotated_frame = mask_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    cv2.imwrite(out_path.absolute().__str__(), annotated_frame)


# %%
# | export
def write_label(detections: sv.Detections, out_path: Path, img_shape: tuple) -> None:
    lines = detections_to_yolo_annotations(
        detections=detections,
        image_shape=img_shape,
    )
    out_path.write_text("\n".join(lines))


# %%
# | export
def get_file_frames(task_list: list[Task]) -> dict[str, list[int]]:
    frame_dict = {}
    for task in tqdm(task_list):
        filename = task.storage_filename
        filename = filename.split("/")[-1]
        file_parts = filename.split("_")
        if len(file_parts) == 2:
            if frame_dict.get(file_parts[0], None) is None:
                frame_dict[file_parts[0]] = []
            frame_dict[file_parts[0]].append(int(file_parts[1].split(".")[0]))
        else:
            name = "_".join(file_parts[0:2])
            if frame_dict.get(name, None) is None:
                frame_dict[name] = []
            frame_dict[name].append(int(file_parts[2].split(".")[0]))
    return frame_dict


# %%
# | export
def extract_task_annotation(
    task_list: list[Task],
    labels: list[str],
    frame_path: Path,
    job_idx: int = 0,
):
    # Make label maps
    label_map = {k: i for i, k in enumerate(labels)}
    label_map_inv = {v: k for k, v in label_map.items()}

    # Collect all frames for annotation
    frame_to_path_dict = {frame.name: frame for frame in frame_path.rglob("*.jpg")}

    # Query labelsutdio api for annotations
    for task in tqdm(task_list, position=job_idx):
        assert task.annotations is not None
        assert task.storage_filename is not None
        filename = task.storage_filename.split("/")[-1]
        out_file_path = frame_to_path_dict[filename]
        out_file_path = out_file_path.parents[1] / "labels" / filename
        out_file_path = out_file_path.with_suffix(".txt")
        if len(task.annotations) == 0:
            # If no annotation is provided then for now skip
            # but in future we will keep the blank frames.txt
            # out_file_path.touch()
            continue

        else:
            print("generating masks")
            masks = []
            classes = []
            scores = []
            img_height = 0
            img_width = 0

            # Collect all annotations for a task
            for anno in task.annotations:
                assert anno["result"] is not None
                for res in anno["result"]:
                    if res["value"].get("brushlabels", None) is not None:
                        curr_label = res["value"]["brushlabels"][0]
                    elif res["value"].get("keypointlabels", None) is not None:
                        curr_label = res["value"]["keypointlabels"][0]
                    else:
                        raise Exception("Unknow label found!")
                    img_width = res["original_width"]
                    img_height = res["original_height"]
                    mask = decode_rle(res["value"]["rle"])
                    mask = np.reshape(mask, [img_height, img_width, 4])[:, :, 3]
                    mask = mask / 255
                    mask = mask.astype(bool)
                    masks.append(mask)
                    classes.append(int(label_map[curr_label]))
                    scores.append(1)

            dets = []
            for i, mask in enumerate(masks):
                detection = sv.Detections.from_transformers(
                    {
                        "scores": t.tensor([scores[i]]),
                        "labels": t.tensor([classes[i]]),
                        "masks": t.tensor(np.array([mask])),
                    },
                    label_map_inv,
                )
                dets.append(detection)
            detections = sv.Detections.merge(dets)
            write_label(detections, out_file_path, (img_height, img_width, 0))
            anno_path = out_file_path.parents[2] / "annotated" / out_file_path.name
            anno_path = anno_path.with_suffix(".jpg")
            write_annotated_image(detections, frame_to_path_dict[filename], anno_path)


# %%
# | export
def extract_annotations(
    client: LabelStudio, project_id: int, out_path: Path, jobs: int = 2
) -> None:
    # Get project
    project = client.projects.get(id=project_id)

    # Get task
    task_pager = client.tasks.list(project=project.id)
    task_list: list[Task] = [task for task in task_pager]

    # Exrtact in parallel
    parallel(task_list, extract_task_annotation, [labels, out_path], jobs)


# %%
create_detection_yolo(labels, local_frames_path, detection_dataset_path)

# %%
extract_annotations(client, project_id, detection_dataset_path, 20)

# %%
# | hide
import nbdev  # noqa

nbdev.nbdev_export()
