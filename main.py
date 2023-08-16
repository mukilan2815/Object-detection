import cv2
import numpy as np

with open("coco.names", 'rt') as f:
    class_names = f.read().strip().splitlines()
config = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weights = "frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weights, config)
net.setInputSize(416, 416)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
image_path = "human/karina-carvalho-fKTKVrNqXQQ-unsplash.jpg"
img = cv2.imread(image_path)
class_ids, confidences, boxes = net.detect(img, confThreshold=0.55)
if len(class_ids) > 0:
    for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
        class_name = class_names[class_id - 1]
        label = f"{class_name.upper()} {confidence:.2f}"
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=4)
        cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
    max_display_height = 800
    aspect_ratio = img.shape[1] / img.shape[0]
    height = max_display_height
    width = int(height * aspect_ratio)
    resized_img = cv2.resize(img, (width, height))
    cv2.imshow("Output window", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No objects detected in the image")
