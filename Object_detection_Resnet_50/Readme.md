# Autonomous_Vehicle_Projects
All caveat of autonomous vehicle stack from vision-planing-to control is covered in this projects. ...is under development. 
CARLA Traffic Video Object Detection 
Object detection pipeline that runs a pretrained TensorFlow SSD-ResNet50-FPN model on recorded CARLA simulator footage (Town
1 traffic scenario) to detect and localize vehicles, pedestrians, traffic lights, and other COCO-class objects frame-by-frame. 
Overview 
This project applies a pretrained 2D object detector to synthetic driving footage generated in CARLA, an open-source autonomous
driving simulator. A traffic scene recorded in CARLA Town 1 is processed frame-by-frame through an SSD-ResNet50-FPN model
(trained on COCO), with detections drawn back onto the video as annotated bounding boxes with class labels and confidence scores. 
The goal is to evaluate how well a standard COCO-pretrained detector performs on simulator-generated imagery, as a stepping
stone toward broader autonomous vehicle perception experiments (e.g., testing detector robustness under simulated sensor
conditions). 
Model 
Architecture: SSD (Single Shot Detector) with ResNet-50 v1 FPN backbone
Model: ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
Training data: COCO 2014
Input resolution: 640×640
Source: TensorFlow Object Detection API Model Zoo 
Pipeline 
The pipeline follows a standard five-stage inference structure, applied per-frame for video: 
Load model — Load the SavedModel once at startup and extract the serving_default inference signature.
Read frame — Read each frame from the input video via OpenCV, convert BGR → RGB.
Preprocess — Convert the frame to a TensorFlow tensor and add a batch dimension.
Run inference — Pass the frame through the model to get raw detection boxes, classes, and scores.
Postprocess & draw — Filter detections by confidence threshold, convert normalized box coordinates to pixel coordinates,
and draw labeled bounding boxes onto the frame. 
Annotated frames are written back out to an output video file. 
Requirements 
tensorflow
opencv-python
numpy
matplotlib

Install with: 
pip install tensorflow opencv-python numpy matplotlib
 Note: GPU acceleration (CUDA-enabled TensorFlow) is strongly recommended. SSD-ResNet50-FPN is compute-heavy —
CPU-only inference on video will be significantly slower than real time. 
Usage 
Download the pretrained model from the TensorFlow Object Detection Model Zoo and place it under a local directory, e.g.:
ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/ └── saved_model/ 
Update the resnet_model_dir and video_path variables in the script to point to your local model and CARLA-recorded
video file.
Run the script: bash python detect_carla_traffic.py 
The annotated output video will be written to the path specified by output_path. 
ConfigurationParameter Description DefaultCONFIDENCE_THRESHOLD Minimum detection score to keep a box 0.5video_path Path to input CARLA-recorded video —output_pathDetected Classes 

The detector uses the standard 80-class COCO label set, including classes relevant to driving scenes such as person, car, truck,
bus, bicycle, motorcycle, and traffic light. 
Notes & Limitations 
The model is trained on real-world COCO imagery, not synthetic/simulated data — detection accuracy on CARLA’s rendered
scenes may differ from real-world performance and is worth evaluating explicitly (domain gap between synthetic and real
imagery).
Processing is currently offline/frame-by-frame, not real-time; inference speed depends on hardware (GPU strongly
recommended for anything approaching real-time throughput).
Bounding boxes reflect a fixed confidence threshold; tune CONFIDENCE_THRESHOLD depending on desired precision/recall
trade-off. 
Future Work 
Benchmark detection performance (mAP, per-class accuracy) on CARLA-rendered footage vs. real-world footage to quantify
the sim-to-real domain gap.
Swap in lighter-weight detectors (e.g., YOLOv8n) to enable real-time inference within the CARLA simulation loop.
Extend to multi-camera or multi-sensor CARLA setups for broader perception evaluation. 
Acknowledgments 
CARLA Simulator — open-source autonomous driving simulator
TensorFlow Object Detection API — pretrained SSD-ResNet50-FPN model
COCO Dataset — training data for the detection model
