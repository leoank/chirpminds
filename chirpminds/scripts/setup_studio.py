"""Automatically set up label studio project."""

import argparse
from pathlib import Path
import os

from label_studio_sdk.client import LabelStudio
import supervision as sv


def get_task_prediction(task, input_data_path: Path):
    detection = sv.DetectionDataset.from_yolo(
        images_directory_path=str(input_data_path.joinpath("train/images")),
        annotations_directory_path=str(input_data_path.joinpath("train/labels")),
        data_yaml_path=str(input_data_path.joinpath("data.yaml"))
    )

    dectections = [v for _,v in detection.annotations.items()]
    breakpoint()

    results = [detection.annotations[task.storage_filename]]
    predictions = []
    for result in results:
        img_width, img_height = result.orig_shape
        boxes = result.boxes.cpu().numpy()
        prediction = {'result': [], 'score': 0.0, 'model_version': "SAM 1"}
        scores = []
        for box, class_id, score in zip(boxes.xywh, boxes.cls, boxes.conf):
            x, y, w, h = box
            prediction['result'].append({
                'from_name': 'label',
                'to_name': 'img',
                'original_width': int(img_width),
                'original_height': int(img_height),
                'image_rotation': 0,
                'value': {
                    'rotation': 0,
                    'rectanglelabels': [result.names[class_id]],
                    'width': w / img_width * 100,
                    'height': h / img_height * 100,
                    'x': (x - 0.5 * w) / img_width * 100,
                    'y': (y - 0.5 * h) / img_height * 100
                },
                'score': float(score),
                'type': 'rectanglelabels',
            })
            scores.append(float(score))
            prediction['score'] = min(scores) if scores else 0.0
            predictions.append(prediction)
    return predictions




def setup_label_studio(input_data_path: Path, api_key: str | None = None):
    # if api_key is None:
    #     API_KEY = os.environ.get("LABEL_STUDIO_KEY")
    # else:
    #     API_KEY = api_key

    API_KEY = "cb6fd0c37537b04e945e969babc98b50b2fb1386"

    client = LabelStudio(api_key=API_KEY)

    yolo_labels = '\n'.join([f'<Label value="{label}"/>' for label in ["chickadee"]])
    label_config = f'''
    <View>
        <Image name="img" value="$image" zoom="true" width="100%" maxWidth="800" brightnessControl="true" contrastControl="true" gammaControl="true" />
        <RectangleLabels name="label" toName="img">
        {yolo_labels}
        </RectangleLabels>
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
        client.predictions.create(task=task["id"], result=predictions['result'], score=predictions['score'], model_version=predictions['model_version'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup studio CLI.")
    parser.add_argument("in_folder", type=str, help="Path to input data folder")
    args = parser.parse_args()
    setup_label_studio(Path(args.in_folder).resolve().absolute())
