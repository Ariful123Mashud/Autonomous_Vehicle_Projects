import pathlib
import tensorflow as tf
import cv2
import numpy as np
import time


gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

## Give model path
resnet_model_dir = pathlib.Path(
    "/home/arif/D_drive/Self_Explore/Autonomous_Vehicle_Stack/AV_Month_works/Month_1/Detection_transfer_learning/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
)
print("Model path:", resnet_model_dir)
print("Model exists:", resnet_model_dir.exists())
print("Loading model...")
resnet_model_dir = resnet_model_dir / "saved_model"

print("Resnet Model path:", resnet_model_dir)
# Load the SavedModel
loaded_model = tf.saved_model.load(str(resnet_model_dir))

print("SavedModel loaded.")
print("Available signatures:", list(loaded_model.signatures.keys()))

# Get the serving signature
resnet_model = loaded_model.signatures["serving_default"]

print("Model loaded successfully.")

COCO_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog",
    # ... (keep your full dict here)
}

CONFIDENCE_THRESHOLD = 0.5

# ---- 2. Open input video (your recorded CARLA footage) ----
video_path = "/home/arif/D_drive/Self_Explore/Autonomous_Vehicle_Stack/AV_Month_works/Month_1/Detection_transfer_learning/Video_files/carla_run_1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {frame_w}x{frame_h} @ {fps:.1f}fps, {total_frames} frames")

# ---- 3. Set up output video writer ----
output_path = "/home/arif/D_drive/Self_Explore/Autonomous_Vehicle_Stack/AV_Month_works/Month_1/Detection_transfer_learning/Video_files/carla_run_1_detected.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

# ---- 4. Frame loop ----
frame_idx = 0
start_time = time.time()

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break  # end of video

    frame_idx += 1

    # --- preprocess ---
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(frame_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    # --- inference ---
    output = resnet_model(input_tensor)

    num_detections = int(output['num_detections'][0])
    detection_boxes = output['detection_boxes'][0, :num_detections].numpy()
    detection_classes = output['detection_classes'][0, :num_detections].numpy().astype(np.int64)
    detection_scores = output['detection_scores'][0, :num_detections].numpy()

    # --- filter by confidence ---
    keep = detection_scores >= CONFIDENCE_THRESHOLD
    detection_boxes = detection_boxes[keep]
    detection_classes = detection_classes[keep]
    detection_scores = detection_scores[keep]

    # --- postprocess: normalized coords -> pixel coords ---
    result_frame = frame_rgb.copy()
    img_h, img_w = result_frame.shape[:2]

    for box, class_id, score in zip(detection_boxes, detection_classes, detection_scores):
        x = int(box[1] * img_w)
        y = int(box[0] * img_h)
        x2 = int(box[3] * img_w)
        y2 = int(box[2] * img_h)

        class_name = COCO_CLASSES.get(int(class_id), "unknown")
        label = f"{class_name}: {score:.2f}"

        cv2.rectangle(result_frame, (x, y), (x2, y2), (0, 255, 0), 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_y = y - 10
        if label_y - text_h < 0:
            label_y = y2 + text_h + 10
        label_x = max(0, min(x, img_w - text_w - 4))

        cv2.rectangle(
            result_frame,
            (label_x, label_y - text_h - baseline),
            (label_x + text_w, label_y + baseline),
            (0, 255, 0), -1
        )
        cv2.putText(
            result_frame, label, (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    # --- write frame back out (convert back to BGR for OpenCV writer) ---
    result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
    out.write(result_frame_bgr)

    if frame_idx % 30 == 0:
        elapsed = time.time() - start_time
        fps_actual = frame_idx / elapsed
        print(f"Frame {frame_idx}/{total_frames}  ({fps_actual:.1f} fps processing)")

cap.release()
out.release()
print(f"Done. Output saved to {output_path}")