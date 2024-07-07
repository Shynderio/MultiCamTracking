# detect/detector.py
import cv2
from abc import ABC, abstractmethod
from ultralytics import YOLO
from .detector import Detector

class Detector(ABC):
    @abstractmethod
    def detect(self, frame):
        pass


# detect/yolo_detector.py

class YOLODetector(Detector):
    def __init__(self, weights):
        self.model = YOLO(weights)

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for r in results:
            for *bbox, conf, cls in r.boxes:
                if int(cls) == 0:  # Assuming '0' is the class ID for 'person'
                    detections.append((bbox, conf))
        return detections
