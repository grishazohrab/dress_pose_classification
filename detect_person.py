import cv2
import numpy as np

import os


net = cv2.dnn.readNet("yolo_model/yolov3.weights", "yolo_model/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def is_person(img_path: str) -> int:
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Preprocess the image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Perform detection
    outs = net.forward(output_layers)

    # Analyze the detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Check if 'person' class is detected
    person_detected = any(class_id == 0 for class_id in class_ids)  # 0 is the class ID for 'person' in YOLO

    if person_detected:
        return 1
    else:
        return 0


if __name__ == '__main__':
    for x in os.listdir("tmp_images/processed_data"):
        file_name = os.listdir(f'tmp_images/processed_data/{x}/1')[0]
        print(f"There is a person in the image "
              f"{is_person(f'tmp_images/processed_data/{x}/1/{file_name}')}")