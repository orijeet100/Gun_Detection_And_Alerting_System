# detect.py
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

def detect_gun(image):
    results = model.predict(image)
    result = results[0]
    annotated_frame = image.copy()

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf > 0.7:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Gun {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated_frame
