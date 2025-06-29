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
# # Tracking experiment video

# %%
# | default_exp tracking

# %%
# | export

import csv
from pathlib import Path

import polars as pl
import supervision as sv
from supervision.detection.utils import box_iou_batch
from tqdm import tqdm
from ultralytics import YOLO

from chirpminds.utils import parallel

# %% [markdown]
# ## Setup labelstudio client


# %%
# | export
# def process_frame(
#     tracker: ByteTrack,
#     bcch_model: YOLO,
#     bcch_onoff_model: YOLO,
#     frame: np.ndarray,
#     frame_id: int,
# ) -> np.ndarray:
#     pass


# %%
# | export
def process_events_csv(in_csv_path: Path):
    df = pl.read_csv(in_csv_path)
    processed = {}
    for frame_id, bird_id, on_antenna, off_antenna, _, _, _, _ in df.iter_rows():
        if processed.get(bird_id, None) is None:
            processed[bird_id] = {
                "enter_frame": frame_id,
                "start_antenna": None if on_antenna == 0 else frame_id,
                "left_antenna": None,
                "left_frame": frame_id,
            }
        else:
            if on_antenna == 1:
                if processed[bird_id]["start_antenna"] is None:
                    processed[bird_id]["start_antenna"] = frame_id
            if off_antenna == 1:
                if processed[bird_id]["left_antenna"] is None:
                    processed[bird_id]["left_antenna"] = (
                        frame_id
                        if processed[bird_id]["start_antenna"] is not None
                        else None
                    )
            processed[bird_id]["left_frame"] = frame_id
    csv_writer = csv.writer(
        in_csv_path.with_suffix(".processed.csv").open("w"),
        delimiter=",",
        quoting=csv.QUOTE_MINIMAL,
    )
    csv_writer.writerow(
        ["bird_id", "enter_frame", "start_antenna", "left_antenna", "left_frame"]
    )
    for k, v in processed.items():
        csv_writer.writerow(
            [
                k,
                v["enter_frame"],
                v["start_antenna"],
                v["left_antenna"],
                v["left_frame"],
            ]
        )


# %%
# | export
def track_video(
    video_path: Path, out_path: Path, bcch_model_path: Path, bcch_onoff_model_path: Path
) -> None:
    # Setup models
    bcch_model = YOLO(bcch_model_path.resolve().__str__())
    bcch_onoff_model = YOLO(bcch_onoff_model_path.resolve().__str__())

    # Setup tracker
    bcch_tracker = sv.ByteTrack(lost_track_buffer=100, minimum_consecutive_frames=3)

    # Setup colors
    bcch_colors = sv.ColorPalette.from_hex(["#20bf55"])
    bcch_onoff_colors = sv.ColorPalette.from_hex(["#ff7845", "#ffe19c"])

    # Setup annotators
    box_annotator = sv.BoxAnnotator(color=bcch_colors)
    onoff_box_annotator = sv.BoxAnnotator(color=bcch_onoff_colors)
    label_annotator = sv.LabelAnnotator(color=bcch_colors)
    trace_annotator = sv.TraceAnnotator(color=bcch_colors, trace_length=150)

    video_info = sv.VideoInfo.from_video_path(video_path.resolve().__str__())
    out_video = out_path.joinpath(video_path.name).resolve().__str__()
    out_csv = out_path.joinpath(video_path.name)
    out_csv = out_csv.with_suffix(".csv")
    csv_writer = csv.writer(out_csv.open("w"), delimiter=",", quoting=csv.QUOTE_MINIMAL)
    frames_generator = sv.get_video_frames_generator(video_path.resolve().__str__())

    with out_csv.open("w") as f:
        csv_writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(
            [
                "frame_id",
                "bcch_id",
                "is_on_antenna",
                "is_off_antenna",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )
        with sv.VideoSink(target_path=out_video, video_info=video_info) as sink:
            for i, frame in enumerate(tqdm(frames_generator)):
                if i > 1000:
                    break
                bcch_res = bcch_model(frame)[0]
                bcch_onoff = bcch_onoff_model(frame)[0]
                csv_rows = []

                # track bcch
                bcch_detections = sv.Detections.from_ultralytics(bcch_res)
                bcch_detections = bcch_tracker.update_with_detections(bcch_detections)

                # Compare bcch detection with on and off antenaa
                bcch_onoff_detections = sv.Detections.from_ultralytics(bcch_onoff)
                for xyxy, _, _, _, tracker_id, _ in bcch_detections:
                    for abab, _, _, class_id, _, _ in bcch_onoff_detections:
                        iou = box_iou_batch(
                            xyxy.reshape((1, 4)), abab.reshape((1, 4))
                        ).reshape((1))[0]
                        print(f"IoU: {iou} for classId: {class_id}")
                        if iou > 0.4:
                            on_antenna = 1 if class_id == 0 else 0
                            off_antenna = 1 if class_id == 1 else 0
                            print(
                                f"bcch {tracker_id}: on: {on_antenna}, off: {off_antenna}"
                            )
                            csv_rows.append(
                                [
                                    i,
                                    tracker_id,
                                    on_antenna,
                                    off_antenna,
                                    *xyxy.reshape((4)).tolist(),
                                ]
                            )
                    print(f"bcch {tracker_id}: on: 0, off: 0")
                    csv_rows.append([i, tracker_id, 0, 0, *xyxy.reshape((4)).tolist()])

                if len(bcch_detections) != 0:
                    # row_writer.writerow([i, "True"])
                    # Apply bounding box to detections on a copy of the frame.
                    annotated_frame = box_annotator.annotate(
                        scene=frame.copy(),
                        detections=bcch_detections,
                    )
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame,
                        detections=bcch_detections,
                    )
                    annotated_frame = onoff_box_annotator.annotate(
                        scene=annotated_frame,
                        detections=bcch_onoff_detections,
                    )
                    annotated_frame = trace_annotator.annotate(
                        scene=annotated_frame,
                        detections=bcch_detections,
                    )
                else:
                    annotated_frame = frame
                csv_writer.writerows(csv_rows)
                sink.write_frame(frame=annotated_frame)


# %%
bcch_model_path = Path(
    "/home/ank/workspace/hub/leoank/chirpminds/main/scratch/trained_models/detect_bcch_cbird_001.pt"
)
bcchonoff_model_path = Path(
    "/home/ank/workspace/hub/leoank/chirpminds/main/scratch/trained_models/detect_bcchonoff_cbird_001_last.pt"
)
video_clip_path = Path(
    "/home/ank/workspace/hub/leoank/chirpminds/main/scratch/raw_data/20230114_F02A.MP4"
)
out_path = Path(
    "/home/ank/workspace/hub/leoank/chirpminds/main/scratch/processed_clips"
)

# %%
track_video(video_clip_path, out_path, bcch_model_path, bcchonoff_model_path)

# %%
bcch_model = YOLO(bcch_model_path)
bcch_model(local_frames[50])

# %%
csv_files = [file for file in out_path.rglob("*.csv")]
process_events_csv(csv_files[0])

# %%
# | hide
import nbdev  # noqa

nbdev.nbdev_export()

# %%
