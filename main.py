import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
from reid import REID
import operator

# Initialize variables
h = 480
w = 640
threshold = 600
exist_ids = set()
final_fuse_id = dict()
images_by_id = dict()
feats = dict()
reid = REID()

def yolobbox2bbox(x, y, w, h):
    """Convert YOLO bbox format to traditional (top, left, bottom, right) format."""
    x1, y1 = x - w/2, y - h/2
    x2, y2 = x + w/2, y + h/2
    return int(x1), int(y1), int(x2), int(y2)

def run_tracker_in_thread(filename, model, output_queue, file_index):
    """Thread function to run YOLOv8 object tracking on a video file or camera stream."""
    video = cv2.VideoCapture(filename)

    while video.isOpened():
        success, frame = video.read()

        if success:
            # Run YOLOv8 tracking on the frame
            results = model.track(frame, persist=True, tracker="/home/shynderio/Python/cam-tracker/bytetrack.yaml", classes=[0])

            # Extract boxes and ids from YOLOv8 results
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
                        if id not in images_by_id:
                            images_by_id[id] = [frame[t:b, l:r]]
                            feats[id] = reid._features(images_by_id[id])
                        else:
                            images_by_id[id].append(frame[t:b, l:r])
                            feats[id] = reid._append_features(feats[id], frame[t:b, l:r])

                        exist_ids.add(id)
                    else:
                        final_fuse_id[id] = id
                        exist_ids.add(id)
                        feats[id] = reid._features([frame[1:2, 1:2]])
                        continue

                    # Calculate distances between current and existing IDs
                    dis = []
                    ids_set = set(ids)
                    o_ids = exist_ids - ids_set

                    for oid in o_ids:
                        if (oid > id):
                            continue
                        tmp = np.mean(reid.compute_distance(feats[id], feats[oid]))
                        if (oid == 1) :
                            print ("tmp: ", tmp)
                        
                        dis.append([oid, tmp])

                    if not dis:
                        final_fuse_id[id] = id
                        continue

                    dis.sort(key=operator.itemgetter(1))
                    print("final_fuse_id: ", final_fuse_id)
                    # Fuse IDs if distance is below threshold
                    if dis[0][1] < threshold:
                        combined_id = dis[0][0]
                        # feats[id] = feats[combined_id]
                        while final_fuse_id[combined_id] != combined_id:
                            combined_id = final_fuse_id[combined_id]

                        feats[combined_id] = reid._cat_features(feats[combined_id], feats[id])
                        images_by_id[combined_id].append(images_by_id[id])

                        final_fuse_id[id] = combined_id
                    else:
                        exist_ids.add(id)
                        final_fuse_id[id] = id

                # Annotate frames with IDs and bounding boxes
                for (id, box) in zip(ids, boxes):
                    t, l, b, r = yolobbox2bbox(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    t = int(max(0, t))
                    l = int(max(0, l))
                    cv2.rectangle(frame, (t, l), (b, r), (255, 0, 0), 2)
                    cv2.putText(frame, str(final_fuse_id[id]), (t, l - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Put annotated frame into the queue for display
                output_queue.put((file_index, frame.copy()))

            # Display the annotated frame (if no objects detected, show original frame)
            else:
                output_queue.put((file_index, frame.copy()))

        # Release video capture if no more frames
        else:
            break

    video.release()

def display_results(output_queue, num_sources):
    """Function to display frames from the output queue."""
    # cv2.namedWindow("Tracking Results", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Tracking Results", 1280, 720)

    active_sources = num_sources
    while active_sources > 0:
        file_index, frame = output_queue.get()
        if frame is None:
            active_sources -= 1
            continue

        cv2.imshow(f"Tracking_Stream_{file_index}", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    # Initialize YOLOv8 model
    model_path = "yolov8n.pt"
    model1 = YOLO(model_path)
    model2 = YOLO(model_path)
    # Define video files or camera streams
    video_file1 = 0  # Webcam (change as needed)
    video_file2 = "https://192.168.1.11:4343/video"  # External camera URL (change as needed)

    # Create a queue to hold processed frames for display
    output_queue = queue.Queue()

    # Create threads for each video source
    tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, output_queue, 1))
    tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, output_queue, 2))

    # Start the tracker threads
    tracker_thread1.start()
    tracker_thread2.start()

    # Start the display thread
    display_thread = threading.Thread(target=display_results, args=(output_queue, 2))
    display_thread.start()

    # Wait for the tracker threads to finish
    tracker_thread1.join()
    tracker_thread2.join()

    # Wait for the display thread to finish
    display_thread.join()

if __name__ == "__main__":
    main()
