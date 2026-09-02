import pathlib
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


## Give model path
resnet_model_dir = pathlib.Path(
    "/home/arif/D_drive/Self_Explore/Autonomous_Vehicle_Stack/AV_Month_works/Month_1/Detection_transfer_learning/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
)

# Point to saved_model
resnet_model_dir = resnet_model_dir / "saved_model"

print("Resnet Model path:", resnet_model_dir)

# Load model: This is where TensorFlow actually loads the trained model from disk.
resnet_model = tf.saved_model.load(str(resnet_model_dir))

# Get inference function
resnet_model = resnet_model.signatures["serving_default"]

print(resnet_model.structured_input_signature)
print(resnet_model.structured_outputs)

## load the image
img_path = "/home/arif/D_drive/Self_Explore/Autonomous_Vehicle_Stack/AV_Month_works/Month_1/Detection_transfer_learning/Images/pic2.jpeg"

img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)

# NumPy → TensorFlow tensor
input_tensor = tf.convert_to_tensor(img)

# Add batch dimension
input_tensor = input_tensor[tf.newaxis, ...]

print("Input shape:", input_tensor.shape)

# Run inference: run the model on the given image
output = resnet_model(input_tensor)

print("Raw output keys:", output.keys())

# Get number of detections
num_detections = int(output['num_detections'][0])
print(num_detections)

# Extract detection results and convert TensorFlow → NumPy
detection_boxes = output['detection_boxes'][0, :num_detections].numpy()
detection_classes = output['detection_classes'][0, :num_detections].numpy()
detection_scores = output['detection_scores'][0, :num_detections].numpy()

# Convert class IDs to integers
detection_classes = detection_classes.astype(np.int64)

print("Number of detections:", num_detections)
print("Classes:", detection_classes)
print("Scores:", detection_scores)
print("Boxes:", detection_boxes)

# Process only the detection tensors
output = {
    key: value[0, :num_detections].numpy()
    for key, value in output.items()
    if key != 'num_detections'
}

# Add number of detections back
output['num_detections'] = num_detections

print("Number of detections:", output['num_detections'])
print("Classes:", output['detection_classes'])
print("Scores:", output['detection_scores'])
print("Boxes:", output['detection_boxes'])


output['detection_classes'] = output['detection_classes'].astype(np.int64)
boxes = []
# COCO class names
COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "TV",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}

for box in output['detection_boxes']:

    new_box = {
        "y": int(box[0] * img.shape[0]),
        "x": int(box[1] * img.shape[1]),
        "y2": int(box[2] * img.shape[0]),
        "x2": int(box[3] * img.shape[1])
    }

    boxes.append(new_box)

output['boxes'] = boxes


result_img = img.copy()
img_h, img_w = result_img.shape[:2]

for box, class_id, score in zip(boxes, detection_classes, detection_scores):
    x, y, x2, y2 = box["x"], box["y"], box["x2"], box["y2"]

    class_name = COCO_CLASSES.get(int(class_id), "unknown")
    label = f"{class_name}: {score:.2f}"

    # Draw the bounding box
    cv2.rectangle(result_img, (x, y), (x2, y2), (0, 255, 0), 2)

    # Measure text size so we can draw a background box behind it
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )

    # Decide label position: above the box normally, but flip below
    # the box if there isn't enough room above (near top of image)
    label_y = y - 10
    if label_y - text_h < 0:
        label_y = y2 + text_h + 10  # draw below the box instead

    # Clamp x so the label never runs off the right edge
    label_x = min(x, img_w - text_w - 4)
    label_x = max(label_x, 0)

    # Solid background rectangle behind text for readability
    cv2.rectangle(
        result_img,
        (label_x, label_y - text_h - baseline),
        (label_x + text_w, label_y + baseline),
        (0, 255, 0),
        -1  # filled
    )

    # Draw text in black on top of the green background (high contrast)
    cv2.putText(
        result_img,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )


# Display
plt.figure(figsize=(12, 8))
plt.imshow(result_img)
plt.axis("off")
plt.show()




















