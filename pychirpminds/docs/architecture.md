# Project technical architecture

## Pipeline

- (optimization) Split source file into multiple chunks for parallel presence detection?

- (enhancement) Generate dataset for fast presence detector
  - Use groudingdino with sam to generate source "bird" dataset for the specific field environment

- (enhancement) Train `fast` presence detector - maybe YOLOv8

- (core) Detect presence
  - detect presence of birds in frames with the trained `fast` detector
  - create timestamps for presence of birds

- (core) Create clips
  - use the timestamps to create clips from source video where birds were detected
    - Make sure clips do not exceed a threshold.
    - split clip into multiple clips when threshold is breached

- (core) Create segmentation masks
  - run groundeddino + sam to segment birds in clips (parallel workers on clips iterable)
  - save masks as videos? (compression for free)

- (maybe core) Covert masks to polygons
  - we might have to convert mask to polygons for some inference tasks

- (core) Motion tracking
  - we might not have to run segmentation on the entire clip
  - we can use traditional motion tracking to track motion after initial segmentation
    - Create patches from mask for each instance and then track patches
    - patches will to help very accuracy of tracking automatically
    - if patches are too far apart then tracking failed

## Inferred Outputs

- 2D Flight path of each detected bird
- Time spent by each detected bird at the feeder and antenna
