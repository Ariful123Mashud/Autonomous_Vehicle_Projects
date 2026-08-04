# Autonomous Vehicle Research Track — Neural MPC + CBF-Safe RL

A 12–18 month, hands-on research and development track building toward a working
**Neural MPC + Control Barrier Function (CBF) Safe RL** control stack for autonomous
vehicles, developed in CARLA and integrated with Autoware.

This repository tracks the full progression: from foundational reading and simulation
setup, through perception/planning/control implementation, to an original research
contribution combining learned control with formally verifiable safety constraints.

---

## Research Direction

**Core question:** How does a Neural MPC + CBF-Safe RL control stack perform — and
degrade — when tested under realistic constraints (sensor faults, degraded
localization, compute/latency limits), rather than idealized simulation conditions?

**Approach:**
- A learned dynamics model augments (not replaces) a classical MPC controller,
  correcting for the gap between nominal vehicle models and real behavior.
- A Control Barrier Function safety layer wraps planning/control output, filtering
  unsafe actions using formally grounded safety constraints rather than heuristics.
- The combined stack is benchmarked against a classical baseline across multiple
  CARLA towns, with explicit fault-injection and robustness testing.
- Where feasible, the resulting safety layer is packaged as an Autoware
  Universe-style module, bridging custom research work into a production AV stack.

---

## Tech Stack

| Component | Tool |
|---|---|
| Simulation | CARLA (0.9.x) |
| Middleware | ROS2 (Jazzy, Ubuntu 24.04) |
| Production AV stack | Autoware (Core/Universe) |
| ML frameworks | PyTorch, Keras/TensorFlow |
| Control | Custom MPC, `mpc_lateral_controller`, `pid_longitudinal_controller` |
| Perception | CenterPoint (LiDAR), YOLOX (camera), custom fine-tuned detectors |
| Deployment | ONNX → TensorRT |
| Environment | Docker, Ansible (Autoware dev-env setup) |
| Hardware | Ubuntu 24.04, NVIDIA GPU (8GB VRAM) |

---

## Roadmap Structure

The project follows a phased roadmap (tracked separately in the accompanying
spreadsheet/Notion import):

1. **Foundations** — CARLA/Autoware setup, ROS2 architecture, Lanelet2 maps,
   localization (NDT + EKF), SLAM mapping, DL foundations (Chollet)
2. **Perception, Planning & Control** — object detection, tracking, prediction,
   behavior/motion planning, classical MPC + PID control (Venturi projects)
3. **Research Core** — learned dynamics models, Neural MPC prototyping, CBF theory,
   CBF safety layer implementation, robustness and fault-injection testing
4. **Capstone** — literature review, formal experiment design, ablation studies,
   results analysis, reproducibility packaging, paper/grant drafting, and an
   Autoware integration pass for the finished pipeline

Reference texts: Liu (*Creating Autonomous Vehicle Systems*), Chollet (*Deep Learning
with Python*), Venturi (*Hands-On Vision and Behavior for Self-Driving Cars*), with
Corke (*Robotics, Vision and Control*) as a supplementary reference for coordinate
transforms and camera geometry.

---

## Key References

Foundational papers this work builds directly on:

- Dosovitskiy et al., *CARLA: An Open Urban Driving Simulator* (CoRL 2017)
- Ames, Xu, Grizzle & Tabuada, *Control Barrier Function Based Quadratic Programs
  for Safety Critical Systems* (IEEE TAC, 2017)
- Lang et al., *PointPillars: Fast Encoders for Object Detection from Point Clouds*
  (CVPR 2019); Yin, Zhou & Krähenbühl, *Center-based 3D Object Detection and
  Tracking* (CVPR 2021)
- Bojarski et al., *End to End Learning for Self-Driving Cars* (NVIDIA, 2016)
- Alahi et al., *Social LSTM: Human Trajectory Prediction in Crowded Spaces*
  (CVPR 2016)

Recent implementation-relevant work (see `/docs/literature/` for the full list with
notes on relevance and reproducibility):
- Learning-augmented MPC via ensemble residual dynamics correction (2026)
- CBF-guided deep RL for on-ramp merging decision-making (IEEE T-ITS, 2025)
- CBF-RL: safety filtering during training, not just deployment (2025)

---

## Repository Structure

```
.
├── docs/
│   ├── roadmap/          # Phase/week tracker (CSV/XLSX)
│   └── literature/       # Paper notes, SOTA references per topic
├── carla_pipeline/
│   ├── perception/       # Detection, tracking, fusion nodes
│   ├── prediction/       # Trajectory prediction models
│   ├── planning/         # Behavior + motion planning
│   └── control/          # Classical MPC, Neural MPC, PID
├── cbf_safety_layer/     # CBF safety filter, standalone + Autoware module variants
├── autoware_integration/ # Bridge config, ported modules, comparison notes
├── experiments/          # Experiment configs, logged results, ablation scripts
├── notebooks/            # Exploratory analysis, model training notebooks
└── docker/               # Dockerfiles for reproducible environment setup
```

---

## Setup

```bash
# Clone
git clone <this-repo-url>
cd <repo-name>

# Build the reproducible environment (Docker)
docker build -t av-research-track -f docker/Dockerfile .

# Or, for local development
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Autoware and CARLA are installed separately per their own official setup guides —
see `docs/setup/autoware-install.md` and `docs/setup/carla-install.md` for the exact
verified command sequences used in this project (Ubuntu 24.04 + Docker + NVIDIA
Container Toolkit).

---

## Status

This project is under active development as part of a self-directed research
track. See `docs/roadmap/` for current phase and week-by-week progress.

---

## Author

Ariful — Assistant Professor, EEE, BITS Pilani WILP.
Research focus: Neural MPC and CBF-Safe Reinforcement Learning for autonomous
vehicles.
