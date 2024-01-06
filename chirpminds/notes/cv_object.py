#%%
import cv2
from random import randint

import numpy as np
from chirpminds.utils import get_default_data_folder

#%%
video_file = get_default_data_folder().joinpath(
    "raw/pilot/feeder_videos/20230114_Feeder02A/MVI_0013_mask.MP4"
)
cap = cv2.VideoCapture(str(video_file.absolute()))
# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
    print("Failed to read video")

bboxes = []
colors = []

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
    bbox = cv2.selectROI("MultiTracker", frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if k == 113:  # q is pressed
        break
print("Selected bounding boxes {}".format(bboxes))

#%%
# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(cv2.TrackerMIL_create(), frame, bbox)

# %%
c