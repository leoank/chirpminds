"""Predict using SAM."""
# %%
import matplotlib.pyplot as plt
import numpy as np
from chirpminds.data.io import yield_frames
from chirpminds.utils import get_default_data_folder
from segment_anything import SamPredictor, sam_model_registry


# %%
def show_mask(mask, ax, random_color=False):  # noqa
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):  # noqa
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):  # noqa
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


# %%
video_file = get_default_data_folder().joinpath(
    "raw/pilot/feeder_videos/20230114_Feeder02A/MVI_0013.MP4"
)
for frame in yield_frames(video_file):
    frame_image = frame.to_image()
    break

# %%
sam = sam_model_registry["vit_h"](
    checkpoint=get_default_data_folder().joinpath("model/sam/sam_vit_h_4b8939.pth")
)
sam.to(device="cuda")
predictor = SamPredictor(sam)

# %%
frame_image_array = np.array(frame_image)

# %%
plt.imshow(frame_image_array)

# %%
predictor.set_image(np.array(frame_image))

# %%
image_embedding = predictor.get_image_embedding().cpu().numpy()
image_embedding

# %%
input_point = np.array([[375, 700], [670, 740]])
input_label = np.array([1, 0])
mask, a, b = predictor.predict(input_point, input_label)

# %%
mask.shape
# %%
for i, (mask, score) in enumerate(zip(mask, a)):
    plt.figure(figsize=(20, 20))
    plt.imshow(frame_image_array)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis("off")
    plt.show()
# %%
