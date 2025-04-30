"""Use the trained model to extract detection positions."""

from pathlib import Path

import yaml
from tqdm import tqdm
from ultralytics import YOLO

# Setup directories

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
