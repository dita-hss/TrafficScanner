from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import torch
from sort import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

capture = cv2.VideoCapture('videos/people.mp4')

model = YOLO('../yolo-weights/yolov8l.pt')

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

mask = cv2.imread("Images/mask2.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limit_up = [103, 161, 296, 161]
limit_down = [527, 489, 735, 489]

total_count_up = set()
total_count_down = set()
while True:
    new_frame_time = time.time()
    success, img = capture.read()
    img_region = cv2.bitwise_and(img, mask)
    results = model(img_region, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
        
            #confidennce
            conf = math.ceil((box.conf[0] * 100)) / 100

            #class name
            cls = int(box.cls[0])
            current_class = class_names[cls]

            if (current_class == "person") and (conf > 0.3):
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    result_tracker = tracker.update(detections)
    cv2.line(img, (limit_up[0], limit_up[1]),  (limit_up[2], limit_up[3]), (0, 0, 255), 5)
    cv2.line(img, (limit_down[0], limit_down[1]),  (limit_down[2], limit_down[3]), (0, 0, 255), 5)
    for result in result_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limit_up[0] < cx < limit_up[2] and limit_up[1] - 15 < cy < limit_up[1] + 15:
            total_count_up.add(id)


        if limit_down[0] < cx < limit_down[2] and limit_down[1] - 15 < cy < limit_down[1] + 15:
            total_count_down.add(id)
            
            


    cvzone.putTextRect(img, f' Count Up: {len(total_count_up)}', (929, 345),
                           scale=2, thickness=3, offset=10)
    cvzone.putTextRect(img, f' Count Down: {len(total_count_down)}', (929, 395),
                           scale=2, thickness=3, offset=10)
        

    cv2.imshow("Image", img)
    cv2.waitKey(0)