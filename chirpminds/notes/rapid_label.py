import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import fiftyone as fo
    return Path, fo


@app.cell
def _(Path):
    dataset_root_path = Path().resolve().parents[1].joinpath("data/dataset/cbird001")
    cbird_train_path = dataset_root_path.joinpath("train/images")
    cbird_valid_path = dataset_root_path.joinpath("valid/images")
    return cbird_train_path, cbird_valid_path, dataset_root_path


@app.cell
def _(cbird_train_path):
    cbird_train_path.__str__() + "/*.jpg"
    return


@app.cell
def _(cbird_train_path, cbird_valid_path, fo):
    cbird_train_fo = fo.Dataset.from_images_patt(cbird_train_path.__str__() + "/*.jpg")
    cbird_valid_fo = fo.Dataset.from_images_patt(cbird_valid_path.__str__() + "/*.jpg")
    return cbird_train_fo, cbird_valid_fo


@app.cell
def _(cbird_train_fo, fo):
    train_sess = fo.launch_app(cbird_train_fo, port=5151, address="0.0.0.0")
    return (train_sess,)


@app.cell
def _():
    from IPython.display import IFrame
    IFrame(src="http://karkinos:5151", width=1500, height=800)
    return (IFrame,)


@app.cell
def _(cbird_train_fo, cbird_valid_fo):
    for _sample in cbird_train_fo:
        _sample.tags.append('train')
        _sample.save()
    for _sample in cbird_valid_fo:
        _sample.tags.append('valid')
        _sample.save()
    return


@app.cell
def _(cbird_train_fo, cbird_valid_fo, fo):
    for _sample in cbird_train_fo:
        _sample['ground_truth'] = fo.Detections()
        _sample.save()
    for _sample in cbird_valid_fo:
        _sample['ground_truth'] = fo.Detections()
        _sample.save()
    return


@app.cell
def _():
    # setup labelstudio params
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
def _(anno_key, cbird_train_fo, label_schema, label_studio_key):
    cbird_train_fo.annotate(anno_key, backend="labelstudio", label_schema=label_schema, api_key=label_studio_key, url="http://localhost:8080")
    return


@app.cell
def _(anno_key, cbird_train_fo):
    cbird_train_fo.get_annotation_info(anno_key)
    return


@app.cell
def _(cbird_train_fo_001, train_sess):
    train_sess.view = cbird_train_fo_001
    return


@app.cell
def _(anno_key, cbird_train_fo):
    cbird_train_fo.load_annotations(anno_key)
    return


@app.cell
def _(cbird_train_fo, cbird_train_path, fo):
    cbird_train_fo.export(export_dir=cbird_train_path.__str__(), dataset_type=fo.types.YOLOv5Dataset, label_field="ground_truth", split="val", classes=["cbird_on_antenna"])
    return


if __name__ == "__main__":
    app.run()
