import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Chirpminds toolkit""")
    return


@app.cell
def _():
    from pathlib import Path
    import fiftyone as fo
    from fiftyone import ViewField as F
    return F, Path, fo


@app.cell
def _(mo):
    file_browser = mo.ui.file_browser(label="Select path to first frame of the video")
    file_browser
    return (file_browser,)


@app.cell
def _(Path, file_browser):
    cbird_frames_path = Path(file_browser.path(0)).resolve().parent
    return (cbird_frames_path,)


@app.cell
def _(cbird_frames_path, fo):
    cbird_dataset = fo.Dataset.from_images_patt(cbird_frames_path.__str__() + "/*.jpg")
    return (cbird_dataset,)


@app.cell
def _(cbird_dataset, fo):
    fo_sess = fo.launch_app(cbird_dataset, address="0.0.0.0", port=5151)
    return (fo_sess,)


@app.cell
def _(fo_sess, mo):
    fo_sess
    mo.iframe("http://karkinos:5151/", height="800px")
    return


@app.cell
def _(cbird_dataset):
    for i, sample in enumerate(cbird_dataset):
        if i <=400:
            sample.tags.append("train")
        else:
            sample.tags.append("val")
        sample.save()
    return i, sample


@app.cell
def _(cbird_dataset):
    ds_tain_view = cbird_dataset.match_tags("train")
    return (ds_tain_view,)


@app.cell
def _(cbird_dataset, fo_sess):
    fo_sess.dataset = cbird_dataset
    return


@app.cell
def _(mo):
    mo.md("""# Setup labeling in LabelStudio""")
    return


@app.cell
def _():
    label_studio_key = "6b6776e528d90bfad8ee5d645066f2cebaa2f57d"
    anno_key = "cbird_train_001"
    label_schema = {
        "ground_truth": {
            "type": "instances",
            "classes": ["cbird_on_antenna"]
        },
        "anohelper": {
            "type": "keypoints",
            "classes": ["cbird_on_antenna"]
        },
    }
    return anno_key, label_schema, label_studio_key


@app.cell
def _(cbird_dataset, fo):
    for _sample in cbird_dataset:
        _sample['ground_truth'] = fo.Detections()
        _sample.save()
    return


@app.cell
def _(anno_key, cbird_dataset, label_schema, label_studio_key):
    cbird_dataset.annotate(anno_key, backend="labelstudio", label_schema=label_schema, api_key=label_studio_key, url="http://localhost:8080")
    return


@app.cell
def _(anno_key, cbird_dataset):
    cbird_dataset.get_annotation_info(anno_key)
    return


@app.cell
def _(mo):
    mo.md("""## Start labeling!""")
    return


@app.cell
def _(mo):
    mo.iframe("http://karkinos:8080/", height="800px")
    return


@app.cell
def _(anno_key, cbird_dataset):
    # load annotations back into the dataset
    cbird_dataset.load_annotations(anno_key)
    return


@app.cell
def _(cbird_dataset, cbird_frames_path, fo):
    # Exporting our annotated dataset
    ds_train_view = cbird_dataset.match_tags("train")
    ds_val_view = cbird_dataset.match_tags("val")
    export_path = cbird_frames_path.parents[1].joinpath("dataset/cbird_001")

    for split, ds_view in [("train", ds_train_view), ("val", ds_val_view)]:
        ds_view.export(
            export_dir=export_path.__str__(),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split=split,
            classes=["cbird_on_antenna"],
        )
    return ds_train_view, ds_val_view, ds_view, export_path, split


@app.cell
def _(mo):
    mo.md("""# Let's train a new model with our annotated dataset!""")
    return


@app.cell
def _():
    from ultralytics import YOLO
    return (YOLO,)


@app.cell
def _(YOLO, export_path):
    model_path = export_path.parents[2].joinpath("models/yolo11x.pt")
    dataset_yaml_path = export_path.joinpath("dataset.yaml")
    model = YOLO(model_path.__str__())
    return dataset_yaml_path, model, model_path


@app.cell
def _(mo):
    mo.iframe("http://karkinos:6006/", height="1000px")
    return


@app.cell
def _(dataset_yaml_path, model):
    mres = model.train(data=dataset_yaml_path.__str__(), epochs=100, imgsz=640, device=[0,1], plots=True)
    return (mres,)


@app.cell
def _(mo):
    mo.md("""# Let's use the model to generate the dataset we need""")
    return


@app.cell
def _(Path, __file__):
    import numpy as np
    import supervision as sv
    import yaml
    from tqdm import tqdm
    import csv

    # Setup directories
    in_video = Path(__file__).parents[2].joinpath("data/MVI_0012.MP4")
    out_video = Path(__file__).parents[2].joinpath("data/done_0012.MP4")
    out_csv = Path(__file__).parents[2].joinpath("data/done_0012.csv")
    trained_model_path = Path(__file__).parents[2].joinpath("runs/detect/train15/weights/best.pt")
    return (
        csv,
        in_video,
        np,
        out_csv,
        out_video,
        sv,
        tqdm,
        trained_model_path,
        yaml,
    )


@app.cell
def _(YOLO, csv, in_video, out_csv, out_video, sv, trained_model_path):
    # predict
    model_trained = YOLO(trained_model_path.resolve().__str__())

    video_info = sv.VideoInfo.from_video_path(in_video)
    frames_generator = sv.get_video_frames_generator(in_video)
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
    return (
        annotated_frame,
        box_annotator,
        csv_file,
        detections,
        frame,
        frames_generator,
        i,
        label_annotator,
        mask_annotator,
        model_trained,
        result,
        row_writer,
        sink,
        video_info,
    )


@app.cell
def _(out_csv):
    import polars as pl

    pl.read_csv(out_csv)
    return (pl,)


if __name__ == "__main__":
    app.run()
