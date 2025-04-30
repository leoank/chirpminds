import argparse
import csv
from pathlib import Path

import supervision as sv
from ultralytics import YOLO


def label_video(trained_model_path: Path, in_video: Path, out_path: Path):
    model_trained = YOLO(trained_model_path.resolve().__str__())
    out_video = out_path.joinpath(in_video.name).resolve().__str__()
    out_csv = out_path.joinpath(f"{in_video.name}.csv")

    video_info = sv.VideoInfo.from_video_path(in_video.resolve().__str__())
    frames_generator = sv.get_video_frames_generator(in_video.resolve().__str__())
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    with sv.VideoSink(target_path=out_video, video_info=video_info) as sink:
        with out_csv.open("w") as csv_file:
            row_writer = csv.writer(
                csv_file, delimiter="\t", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            row_writer.writerow(["frame_id", "is_bird_detected_near_antenna"])
            for i, frame in enumerate(frames_generator):
                result = model_trained(frame)[0]
                # Parse the result into the detections data model.
                detections = sv.Detections.from_ultralytics(result)
                if len(detections) != 0:
                    row_writer.writerow([i, "True"])
                    # Apply bounding box to detections on a copy of the frame.
                    annotated_frame = box_annotator.annotate(
                        scene=frame.copy(), detections=detections
                    )
                    annotated_frame = mask_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                else:
                    annotated_frame = frame
                sink.write_frame(frame=annotated_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label video")
    parser.add_argument("model", type=str, help="Path to trained model")
    parser.add_argument("in_video", type=str, help="Path to input video")
    parser.add_argument("out_path", type=str, help="Path to output data folder")
    args = parser.parse_args()
    label_video(Path(args.model), Path(args.in_video), Path(args.out_path))
