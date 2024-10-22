# Simple Diffusion Pose Gen

A quick attempt to utilize [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) (SD) generative models (v1.5 & XL) and MediaPipe's [Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) vision models to generate 3D human poses from an AI prompt.

| ![SD Pose Gen](img/sd_pose_gen.png) |
| :--: |
| Generating a pose with a prompt using SD v1.5 |

## Python Depedencies

* torch
* diffusers
* huggingface-hub
* mediapipe
* pyside6 (>=6.7)

## Start-up SD Model Settings

![SD Settings](img/sd_settings.png)

* *Stable Diffusion v1.5* has fewer model parameters and therefore requires less memory whereas *Stable Diffusion XL* generates better-quality images with more realistic human poses.
* [Hyper-SD](https://hyper-sd.github.io/) (a SD inference acceleration algorithm) Steps: *2*, *4*, or *8*, trade-off between speed (fewer bigger steps) and quality (more smaller steps).
* PyTorch device for running SD inference (if available): *CPU*, *CUDA*, or *MPS* (Apple Silicon).