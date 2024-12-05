"""Inference with trained models."""

# pyright: basic
from pathlib import Path
import numpy as np
import supervision as sv
from ultralytics import YOLO


def infer_yolo(model_wt: Path, in_vid: Path, out_dir: Path) -> sv.ByteTrack:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_wt.resolve()))
    tracker = sv.ByteTrack(lost_track_buffer=300)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator(trace_length=500)

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        results = model.predict(frame)[0]  # pyright: ignore
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]  # pyright: ignore

        annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        return annotated_frame

    sv.process_video(
        source_path=str(in_vid.resolve()),
        target_path=str(out_dir.joinpath(in_vid.name).resolve()),
        callback=callback,
    )
    return tracker
