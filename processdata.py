import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
from reid import REID
import operator
import argparse
import os

h = 480
w = 640

images_by_id = dict()
label_feats = dict()
def yolobbox2bbox(x, y, w, h):
    """Convert YOLO bbox format to traditional (top, left, bottom, right) format."""
    x1, y1 = x - w/2, y - h/2
    x2, y2 = x + w/2, y + h/2
    return int(x1), int(y1), int(x2), int(y2)


reid = REID('resnet50')
model = YOLO("yolov8n.pt")

def ExtractFeatures(filename, extension):
    source = 'data/' + filename + '.' + extension
    video = cv2.VideoCapture(source)
    while video.isOpened():
        success, frame = video.read()
        if success:
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], device=0)
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                ids = results[0].boxes.id.cpu().numpy()
                for id, box in zip(ids, boxes):
                    t, l, b, r = yolobbox2bbox(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    t = int(max(0, t))
                    l = int(max(0, l))
                    b = int(min(h, b))
                    r = int(min(w, r))
                    if t < h and l < w and b > 0 and r > 0:
                        if filename not in images_by_id:
                            images_by_id[id] = [frame[t:b, l:r]]
                            label_feats[filename] = reid._features(images_by_id[id])
                        else:
                            images_by_id[id].append(frame[t:b, l:r])
                            label_feats[filename] = reid._append_features(label_feats[filename], frame[t:b, l:r])
        else:
            break


# def ExtractImages(source):
#     for person_dir in os.listdir(source):
#         for cam_dir in os.listdir(os.path.join(source, person_dir)):
#             for file in os.listdir(os.path.join(source, person_dir, cam_dir)):
#                 if file.endswith('.jpg'):
#                     img = cv2.imread(os.path.join(source, person_dir, cam_dir, file))
#                     if person_dir not in images_by_id:
#                         images_by_id[person_dir] = [img]
#                         label_feats[person_dir] = reid._features(images_by_id[person_dir])
#                     else:
#                         images_by_id[person_dir].append(img)
#                         label_feats[person_dir] = reid._append_features(label_feats[person_dir], img)       

ExtractFeatures('thinh', 'mp4')
ExtractFeatures('thinh', 'avi')
# ExtractFeatures('vo')
ExtractFeatures('sam', 'mp4')
ExtractFeatures('sam', 'avi')
# ExtractImages('cropped_images')

np.save('features_dict.npy', label_feats)