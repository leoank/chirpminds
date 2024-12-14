"""Automatically set up label studio project."""

import argparse
from pathlib import Path
import os

import numpy as np
from label_studio_sdk.client import LabelStudio
import supervision as sv


def get_task_prediction(task, input_data_path: Path):
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=str(input_data_path.joinpath("train/images")),
        annotations_directory_path=str(input_data_path.joinpath("train/labels")),
        data_yaml_path=str(input_data_path.joinpath("data.yaml"))
    )

    detections = dataset.annotations[task.storage_filename]
    predictions = []
    polygons = [ sv.mask_to_polygons(m) for m in detections.mask ]
    prediction = {'result': [], 'score': 1.0, 'model_version': "SAM 1"}
    scores = []
    for polygon in polygons:
        polygon = polygon[0].tolist()
        prediction['result'].append({
            'from_name': 'label',
            'to_name': 'img',
            'image_rotation': 0,
            "original_width": 1920,
            "original_height": 1080,
            'value': {
                "points": [[(p[0]/1920)*100, (p[1]/1080)*100] for p in polygon],
                "polygonlabels": ["chickadee"]
            },
            'score': 1.0,
            'type': 'polygonlabels',
        })
    predictions.append(prediction)
    return predictions




def setup_label_studio(input_data_path: Path, api_key: str | None = None):
    # if api_key is None:
    #     API_KEY = os.environ.get("LABEL_STUDIO_KEY")
    # else:
    #     API_KEY = api_key

    API_KEY = "cb6fd0c37537b04e945e969babc98b50b2fb1386"

    client = LabelStudio(api_key=API_KEY)

    yolo_labels = '\n'.join([f'<Label value="{label}" background="red"/>' for label in ["chickadee"]])
    label_config = f'''
    <View>
        <Image name="img" value="$image" zoom="true" width="100%" maxWidth="800" brightnessControl="true" contrastControl="true" gammaControl="true" />
        <PolygonLabels name="label" toName="img" strokeWidth="1" pointSize="small" opacity="0.6">
            {yolo_labels}
        </PolygonLabels>
    </View>'''

    # Create project
    project = client.projects.create(
    title="Refine Chikadee Segmentation",
    description="Add desc later.",
    label_config=label_config
    )

    # Add storage
    storage = client.import_storage.local.create(
    title="Local storage",
    description="Add later.",
    project=project.id,
    regex_filter=".*jpg",
    use_blob_urls=True,
    path=str(input_data_path.joinpath("train/images")),
    )

    # Sync storage
    client.import_storage.local.sync(id=storage.id)

    # Get task
    tasks = client.tasks.list(project=project.id)
    for i, task in enumerate(tasks):
        predictions = get_task_prediction(task, input_data_path)[0]
        client.predictions.create(task=task.id, result=predictions['result'], score=predictions['score'], model_version=predictions['model_version'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup studio CLI.")
    parser.add_argument("in_folder", type=str, help="Path to input data folder")
    args = parser.parse_args()
    setup_label_studio(Path(args.in_folder).resolve().absolute())
