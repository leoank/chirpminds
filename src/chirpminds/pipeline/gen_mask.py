"""Generate mask."""

import groundingdino.datasets.transforms as gdino_transforms
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from PIL import Image

# from PIL import Image, ImageDraw, ImageFont
from segment_anything import SamPredictor, sam_model_registry

from chirpminds.utils import get_default_data_folder


def load_image(image_pil):
    transform = gdino_transforms.Compose(
        [
            gdino_transforms.RandomResize([800], max_size=1333),
            gdino_transforms.ToTensor(),
            gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(
    model,
    image,
    caption,
    box_threshold,
    text_threshold=None,
    with_logits=True,
    cpu_only=False,
    token_spans=None,
):
    assert (
        text_threshold is not None or token_spans is not None
    ), "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption), token_span=token_spans
        ).to(
            image.device
        )  # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for token_span, logit_phr in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = " ".join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend(
                    [phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num]
                )
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


dino_config = get_default_data_folder().parent.joinpath("notes/dino_config.py")
model_check_path = get_default_data_folder().joinpath(
    "model/groundingdino/groundingdino_swint_ogc.pth"
)
gdino = load_model(dino_config, model_check_path, cpu_only=False)
sam = sam_model_registry["vit_h"](
    checkpoint=get_default_data_folder().joinpath("model/sam/sam_vit_h_4b8939.pth")
)
sam.to(device="cuda")


def gen_mask(frame_image: Image) -> tuple[np.ndarray, np.ndarray]:
    """Generate mask for a frame image."""
    image_pil, image = load_image(frame_image)
    boxes_filt, pred_phrases = get_grounding_output(
        gdino,
        image,
        "bird",
        0.3,
        0.28,
        False,
        token_spans=None,
    )
    predictor = SamPredictor(sam)
    predictor.set_image(np.array(frame_image))
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_filt, frame_image.size
    ).to(predictor.device)
    masks, scores, _ = predictor.predict_torch(
        None, None, boxes=transformed_boxes, multimask_output=False
    )
    for i, score in enumerate(scores):
        if score < 0.8:
            masks[i] = False
    return (masks.cpu().numpy(), transformed_boxes)
