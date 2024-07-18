import cv2
import os
from ultralytics import YOLO
from reid import REID

# Initialize YOLO and REID models
reid = REID('resnet50')
model = YOLO("yolov8n.pt")
h = 480
w = 640

# Convert YOLO bbox format to traditional (top, left, bottom, right) format
def yolobbox2bbox(x, y, w, h):
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return int(x1), int(y1), int(x2), int(y2)

# Save bounding box image to specified path
def save_bbox_image(image, bbox, save_path, object_id):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    bbox_image_path = os.path.join(save_path, f"{object_id}.png")
    cv2.imwrite(bbox_image_path, bbox)

# Function to process a single image
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image from {image_path}")
        return

    results = model(image, classes=[0], device=0, conf=0.95, show=True)

    for result in results:
        boxes = result.boxes.xywh.cpu()
        # ids = result.boxes.id.cpu().numpy()
        # if result.boxes
        for box in  boxes:
            # t, l, b, r = yolobbox2bbox(box[0], box[1], box[2], box[3])
            t, l, b, r = yolobbox2bbox(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            t = int(max(0, t))
            l = int(max(0, l))
            b = int(min(h, b))
            r = int(min(w, r))

            bbox = image[t:b, l:r]
            print(t, l, b, r)
            save_path = os.path.join('bounding_boxes', os.path.splitext(os.path.basename(image_path))[0])
            save_bbox_image(image, bbox, save_path, 1)

# Function to process all images in a directory
def process_images_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            process_image(image_path)

# Example usage
process_images_in_directory("data/video/")
