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
# | default_exp labelstudio

# %%
# | export

from pathlib import Path

import numpy as np
import supervision as sv
from label_studio_sdk.client import LabelStudio
from label_studio_sdk.converter.brush import decode_rle
from label_studio_sdk.types import Task
from supervision import Detections
from tqdm import tqdm

from chirpminds.utils import parallel

# %% [markdown]
# ## Setup labelstudio client


# %%
# | export
def setup_project(
    client: LabelStudio,
    project_title: str,
    labels: list[str],
    frames_path: Path,
    model_url: str = "http://localhost:9090",
    img_ext: str = "jpg",
):
    # setup labels
    colors = ["#ffbe0b", "#fb5607", "#ff006e", "#8338ec", "#3a86ff"]
    annotation_labels = "\n".join(
        [
            f'<Label value="{label}" background="{colors[i]}"/>'
            for i, label in enumerate(labels)
        ]
    )

    label_config = f"""
    <View>
        <Image name="img" value="$image" zoom="true" width="100%" maxWidth="800" brightnessControl="true" contrastControl="true" gammaControl="true" />
        <Hypertext name="brush" className="help">
          <span> BrushLabels for manual labelling </span>
         </Hypertext>
        <BrushLabels name="label" toName="img" opacity="0.6">
            {annotation_labels}
        </BrushLabels>
        <Hypertext name="keypoints" className="help">
          <span> Keypoints for auto labelling </span>
         </Hypertext>
        <KeyPointLabels name="tag2" toName="img" smart="true">
            {annotation_labels}
        </KeyPointLabels>
    </View>"""

    # create project
    project = client.projects.create(
        title=project_title, description="Video annotation", label_config=label_config
    )

    # add storage
    storage = client.import_storage.local.create(
        title=f"{project_title} storage",
        description="Storage for annoataion task.",
        project=project.id,
        regex_filter=f".*{img_ext}",
        use_blob_urls=True,
        path=str(frames_path.resolve()),
    )

    # Sync storage
    client.import_storage.local.sync(id=storage.id)

    # Add ml model for interactive predictions
    client.ml.create(
        title=f"{project_title}_ml_model",
        description="Interactive annoataion",
        url=model_url,
        project=project.id,
        is_interactive=True,
    )

    return project


# %% [markdown]
# ## Setting up labels
#
# bcch
# antenna
#
# During annotation, use the region merge feature to create this category
# bcch_on_antenna
#
# Apply this label to bird region by copying bird region
# bcch_off_antenna

# %%
api_key = "b0429a22dacfad6c4fe937b8d9ddcf7b8dce7de7"
base_url = "http://karkinos:8080"
model_url = "http://10.13.84.1:9090"
labels = ["bcch", "antenna", "bcch_on_aantenna", "bcch_off_antenna"]
# Path inside the container
frames_path = Path("/datastore/frames")
client = LabelStudio(api_key=api_key, base_url=base_url)


# %%
project = setup_project(
    client, "Annoate chickadee variety", labels, frames_path, model_url
)

# %%
client.projects.list()

# %%
client.projects.delete(9)

# %% [markdown]
# ## Get annotations back from labelstudio

# %%
from IPython.display import IFrame

IFrame("http://karkinos:8080", width=1200, height=700)


# %%
def write_detection_yolo(detection: Detections, out_path: Path) -> None:
    pass


# %%
def extract_task_annotation(
    task_list: list[Task], labels: list[str], out_path: Path, job_idx: int = 0
):
    label_map = {k: i for i, k in enumerate(labels)}
    label_map_inv = {v: k for k, v in label_map.items()}
    for task in tqdm(task_list, position=job_idx):
        assert task.annotations is not None
        if len(task.annotations) == 0:
            break
        else:
            print("generating masks")
            dets = []
            for anno in task.annotations:
                assert anno["result"] is not None
                for res in anno["result"]:
                    mask = decode_rle(res["value"]["rle"])
                    mask = np.reshape(mask, [1080, 1920, 4])[:, :, 3]
                    mask = mask / 255
                    mask = mask.astype(bool)
                    mask = t.tensor(np.array([mask]))
                    labels = t.tensor(np.array([0]))
                    scores = t.tensor(np.array([1]))
                    det = sv.Detections.from_transformers(
                        {"scores": scores, "labels": labels, "masks": mask},
                        {0: "cbird_on_antenna"},
                    )
                    dets.append(det)
            detections = sv.Detections.merge(dets)


# %%
def extract_annotations(client: LabelStudio, project_id: int, out_path: Path) -> None:
    # Get project
    project = client.projects.get(id=project_id)

    # Get task
    task_pager = client.tasks.list(project=project.id)
    task_list: list[Task] = [task for task in task_pager]

    # Exrtact in parallel
    parallel(task_list, extract_task_annotation, [out_path])


# %%
# | hide
import nbdev  # noqa

nbdev.nbdev_export()
