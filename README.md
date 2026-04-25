# Simple Diffusion Pose Gen

A quick attempt to utilize [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) (SD) image generative models (v1.5 & XL) and MediaPipe's [Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) vision models to generate 3D human poses from an AI prompt.

## News

* 2026-04-25: Updated project to build/run with [Pixi](https://pixi.prefix.dev).
* 2025-06-22: Minor updates on docs and dependencies.
* 2024-11-12: Added a Godot full-body IK pose solver.
* 2024-10-24: Initial implementation.

## Installation

1. Clone repo: `git clone https://github.com/jerenchen/simple-diffusion-pose-gen.git`.
2. ***(Skip if using Pixi)*** Install Python dependency:
    1. Change dir into `python` and (optionally) use a virtual enivronment (e.g. conda).
    2. Install Python depedencies: `pip install -r requirements.txt`.
3. Download MediaPipe Pose Landmarker:
   - `pixi run download-landmaker`, or
   - Manually download [Pose Landmarker (Full)](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task) and save the file under `python/tasks`.

## Python Pose Gen + Godot IK Solver

1. Initialize the Python pose-gen service:
   * `pixi run posegen-server -- --base sd15 --steps 8`
   * Or, without Pixi: Inside dir `python`, run `python posegen.py --base sd15 --steps 8`.
2. Open and run `projec.godot` inside dir `godot` with [Godot Engine](https://godotengine.org/).
3. Enter a prompt to generate a pose.

| ![SD Godot Pose Gen](img/sd_godot_pose_gen.png) |
| :---: |
| Generating a pose using prompt *"A basketball player making a 3-pointer jump shot"* |

## Standalone Python Pose Gen

> NOTE: The standalone demo requires PySide6 >= v6.7.

`pixi run diffusion-posegen`, and then enter a prompt to generate a pose.

(Without Pixi: Inside dir `python`, run: `python simple-diffusion-pose-gen.py`)

| ![SD Settings](img/sd_settings.png) |
| :---: |
| ![SD Pose Gen](img/sd_pose_gen.png) |

## Stable Diffusion Model Settings

* *SD15* has fewer model parameters and therefore requires less memory whereas *SDXL* could generate images with higher fidelity.
* [Hyper-SD](https://hyper-sd.github.io/) (a SD inference acceleration technique) Steps: *2*, *4*, or *8*, trade-off between speed (fewer bigger steps) and quality (more smaller steps).
* PyTorch device for running SD inference (if available): *CPU*, *CUDA*, or *MPS* (Apple Silicon).

## Acknowledgement

* [Godot + Python Communication with UDP](https://github.com/trflorian/godot-python-comm) by [Florian Trautweiler](https://github.com/trflorian).
* [Open 3D Mannequine](https://github.com/gdquest-demos/godot-3d-mannequin) by [GDQuest](https://github.com/gdquest-demos).