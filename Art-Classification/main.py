import cv2
import numpy as np
import os
import tempfile
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load YOLO
net = cv2.dnn.readNet("yolo-coco/yolov3.weights", "yolo-coco/yolov3.cfg")
classes = []
with open("yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

# Adjusting to handle the returned scalar values
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify(error="No image file found"), 400

    image_file = request.files['image']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image_path = temp_file.name
        image_file.save(image_path)

    img = cv2.imread(image_path)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    tags = set()

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                tags.add(classes[class_id])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
    detected_tags = [classes[class_ids[i]] for i in range(len(boxes)) if i in indexes]

    # Clean up the temporary file
    os.remove(image_path)

    return jsonify(tags=list(tags))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
