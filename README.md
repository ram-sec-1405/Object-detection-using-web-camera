# Object-detection-using-web-camera

<H3>Name:RAMPRASATH.R </H3>
<H3>Register no:212223220086</H3>


# AIM:
To perform real-time object detection using a trained YOLO v4 model through your laptop camera.
# Algorithm:

 Initialize the webcam using OpenCV. Load the YOLOv4 pre-trained model (yolov4.weights) and
 configuration (yolov4.cfg). Load the COCO class labels (coco.names) and assign random colors for
 visualization. Capture frames continuously from the webcam. Preprocess each frame by creating a
 blob and pass it through the YOLOv4 network. Extract detected object bounding boxes, class IDs,
 and confidences. Apply Non-Maximum Suppression (NMS) to remove overlapping boxes. Draw
 bounding boxes and labels for detected objects on the frame. Display the frames inline in Jupyter
 Notebook using matplotlib. Stop detection by pressing the “Stop Detection” button in the
 notebook
# PROGRAM:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
from threading import Thread
```
```


# Load YOLOv4
net = cv2.dnn.readNet(r"yolov4.weights",
                      r"yolov4 (1).cfg")

```
```
# Load COCO classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

```
```
# Get output layers
layer_names = net.getLayerNames()
outs = net.getUnconnectedOutLayers()
if len(outs.shape) == 1:
    output_layers = [layer_names[i - 1] for i in outs]
else:
    output_layers = [layer_names[i[0] - 1] for i in outs]

```
```
# Start webcam
cap = cv2.VideoCapture(0)
```
```
# Create a stop button
stop_button = widgets.Button(description="Stop Detection", button_style='danger')
display(stop_button)
stop_flag = False

def stop_detection(b):
    global stop_flag
    stop_flag = True

stop_button.on_click(stop_detection)

def detect_objects():
    global stop_flag
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, channels = frame.shape

        # Prepare blob and forward pass
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []

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

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indexes.flatten() if len(indexes) > 0 else []:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display frame inline
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        clear_output(wait=True)
        plt.imshow(frame_rgb)
        plt.axis('off')
        display(plt.gcf())

# Run detection in a separate thread so the stop button works
thread = Thread(target=detect_objects)
thread.start()

```
# OUTPUT:

<img width="647" height="501" alt="image" src="https://github.com/user-attachments/assets/6aa2bf19-9ded-4b98-819b-2dcb104c97c3" />

# Result:
The webcam captures live video frames.
 YOLOv4 detects objects like person, chair, laptop, bottle, etc. in real-time.
 Detected objects are highlighted with bounding boxes and class labels.
 Frames are displayed inline in the Jupyter Notebook.
 Detection stops when the “Stop Detection” button is pressed.
 
