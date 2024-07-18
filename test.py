import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
from reid import REID
import operator
import argparse

# Initialize variables
h = 480
w = 640
threshold = 350
exist_ids = set()
final_fuse_id = dict()
images_by_id = dict()
feats = dict()
label_feats = dict()
label_feats = np.load('features_dict.npy', allow_pickle=True).item()

model = YOLO("yolov8n.pt")

def yolobbox2bbox(x, y, w, h):
    """Convert YOLO bbox format to traditional (top, left, bottom, right) format."""
    x1, y1 = x - w/2, y - h/2
    x2, y2 = x + w/2, y + h/2
    return int(x1), int(y1), int(x2), int(y2)


def run_tracker_in_thread(model, video_cap):
    """Thread function to run YOLOv8 object tracking on a video file or camera stream."""
    video = video_cap
    reid = REID('resnet50')

    # label_feats[1] = reid._features([cv2.imread("img.jpg")])
    # label_feats[1] = reid._features([cv2.imread("img2.jpg")])
    while video.isOpened():
        success, frame = video.read()
        if success:

            # Run YOLOv8 tracking on the frame
            results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[0], device=0)
            
            # Extract boxes and ids from YOLOv8 results
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                ids = results[0].boxes.id.cpu().numpy()
                dis = []
                for id, box in zip(ids, boxes):
                    t, l, b, r = yolobbox2bbox(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    t = int(max(0, t))
                    l = int(max(0, l))
                    b = int(min(h, b))
                    r = int(min(w, r))

                    if t < h and l < w and b > 0 and r > 0:
                        if id not in images_by_id:
                            images_by_id[id] = [frame[t:b, l:r]]
                            feats[id] = reid._features(images_by_id[id][:100])
                        else:
                            images_by_id[id].append(frame[t:b, l:r])
                            feats[id] = reid._append_features(feats[id], frame[t:b, l:r])

                        exist_ids.add(id)
                    else:
                        # final_fuse_id[id] = id
                        exist_ids.add(id)
                        feats[id] = reid._features([frame[1:2, 1:2]])
                        continue

                
                    for label in label_feats.keys():
                        tmp = np.mean(reid.compute_distance(feats[id], label_feats[label]))
                        print("tmp: ", tmp)
                        dis.append([id, label, tmp])


                # Annotate frames with IDs and bounding boxes
                dis.sort(key=operator.itemgetter(2))
                # print("dis: ", dis)
                selected_id = set()
                selected_label = set()
                for id, label, tmp in dis:
                    if id not in selected_id and label not in selected_label:
                        selected_id.add(id)
                        selected_label.add(label)
                        if tmp < threshold:
                            final_fuse_id[id] = label
                # for id, tmp in dis:
                #     if oid not in selected.values():
                #         selected[id] = oid
                #         # selected[id] = oid
                combined_id = -1
                # if dis[0][1] < threshold:
                #     # if dis[0][0] in final_fuse_id.keys():
                #     #     combined_id = final_fuse_id[dis[0][0]]
                #     print("dis: ", dis[0][0])
                #     final_fuse_id[dis[0][0]] = 1
                    # combined_id = dis[0][0]
                    
                print("final_fuse_id: ", final_fuse_id)
                
                for (id, box) in zip(ids, boxes):
                    t, l, b, r = yolobbox2bbox(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    t = int(max(0, t))
                    l = int(max(0, l))
                    cv2.rectangle(frame, (t, l), (b, r), (255, 0, 0), 2)
                    if (id in final_fuse_id.keys()):
                        cv2.putText(frame, str(final_fuse_id[id]), (t, l - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, 'unknown', (t, l - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                # Put annotated frame into the queue for display
                # output_queue.put((file_index, frame.copy()))

            # Display the annotated frame (if no objects detected, show original frame)
            # else:
                # output_queue.put((file_index, frame.copy()))

        # Release video capture if no more frames
        else:
            break

    # video.release()

filenames = [0, '']


videos = [cv2.VideoCapture(url) for url in filenames]

threads = []
for cap in videos:
    thread = threading.Thread(target=run_tracker_in_thread, args=(model,  cap))

for cap in videos:
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()