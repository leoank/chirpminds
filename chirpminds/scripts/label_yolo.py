"""Use the trained model to label images."""

from pathlib import Path

import yaml
from tqdm import tqdm
from ultralytics import YOLO

# Setup directories
images_path = Path(__file__).parents[2].joinpath("data/inoutdata/images")
images_path.mkdir(parents=True, exist_ok=True)
labels_path = Path(__file__).parents[2].joinpath("data/inoutdata/labels")
labels_path.mkdir(parents=True, exist_ok=True)
plot_path = Path(__file__).parents[2].joinpath("data/inoutdata/plot")
plot_path.mkdir(parents=True, exist_ok=True)

# predict
model = YOLO("../../runs/segment/train2/weights/best.pt")


# Save labels
results = model(images_path, stream=True)
for i, result in enumerate(tqdm(results)):
    result.save_txt(
        filename=labels_path.joinpath(result.path.name.replace("jpg", "txt"))
    )
    result.save(filename=plot_path.joinpath(result.path.name))

# write data.yml
data_dict = {
    "names": ["chickadee", "chickadee_in", "chickadee_out"],
    "nc": 3,
    "train": str(images_path.resolve()),
}
with images_path.parent.joinpath("data.yaml").open("w") as f:
    yaml.dump(data_dict, f)
