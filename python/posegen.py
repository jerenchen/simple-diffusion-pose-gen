from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
import mediapipe as mp
from mediapipe import solutions as mps
from mediapipe.framework.formats import landmark_pb2

import os
import torch
import numpy
from PIL import Image

import socket
import json
import string
import argparse
import logging


log = logging.getLogger('SDPG')
log.addHandler(logging.StreamHandler())
logging.basicConfig(filename='sdpg.log', level=logging.DEBUG)

SDPG_SDMODELS = {
  'SDXL': "stabilityai/stable-diffusion-xl-base-1.0",
  'SD15': "stable-diffusion-v1-5/stable-diffusion-v1-5"
}
SDPG_BASESD = 'SD15'
SDPG_STEPS = 8
SDPG_DEVICE = 'cpu'
SDPG_IMGSIZE = 512
SDPG_TASKPATH = './tasks/pose_landmarker_full.task'
SDPG_PORT = 42442

class SimpleDiffusionPoseGen:

  def __init__(self, steps: int = SDPG_STEPS, base: str = SDPG_BASESD, device: str = SDPG_DEVICE):

    log.info("Initializing base stable diffusion model {} with {} device...".format(base, device))
    dtype = torch.float32 if device=='cpu' else torch.float16
    pipe = DiffusionPipeline.from_pretrained(SDPG_SDMODELS[base], variant="fp16",
      torch_dtype=dtype, use_safetensors=True)
    if device in ['cuda', 'mps']:
      pipe.to(device)

    log.info("Setting up unified LoRA & scheduler for Hyper-SD with {} steps...".format(steps))
    ckpt = f"Hyper-{base}-{steps}steps-lora.safetensors"
    pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", ckpt))
    pipe.fuse_lora()
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    log.info("Setting up Mediapipe for pose landmarker model {}...".format(SDPG_TASKPATH))
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    self._sd_landmark_options = PoseLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=SDPG_TASKPATH),
      running_mode=VisionRunningMode.IMAGE
    )

    self._sd_pipe = pipe
    self._sd_steps = steps
    self._img_size = 1024 if base=='SDXL' else SDPG_IMGSIZE
  
  def generate_image(self, prompt: str, seed: int):
    '''
     Generate image via SD
    '''
    image = self._sd_pipe(
      prompt=prompt+', realistic human pose, high quality, best quality',
      negative_prompt='more than 2 legs, more than 2 arms, partial body, blurry, low quality',
      eta=1.0,
      guidance_scale=0,
      num_inference_steps=self._sd_steps,
      generator=torch.manual_seed(seed),
      width=self._img_size,
      height=self._img_size
    ).images[0]

    if self._img_size > SDPG_IMGSIZE:
      image = image.resize((SDPG_IMGSIZE, SDPG_IMGSIZE))

    return image

  def detect_landmarks(self, image):
    '''
    Estimate pose landmarks using Google MediaPipe
    '''
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    landmarker = PoseLandmarker.create_from_options(self._sd_landmark_options)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy.asarray(image))
    return landmarker.detect(img)

  def annotate_landmarks(self, image, poses):
    '''
    Draw pose landmarks on the image
    '''
    I = numpy.asarray(image).copy()
    # Loop through the detected poses to visualize.
    for P in poses.pose_landmarks:
      # Draw the pose landmarks.
      N = landmark_pb2.NormalizedLandmarkList()
      N.landmark.extend([landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z) for p in P])
      mps.drawing_utils.draw_landmarks(
        I,
        N,
        mps.pose.POSE_CONNECTIONS,
        mps.drawing_styles.get_default_pose_landmarks_style())
    return Image.fromarray(numpy.uint8(I))

  def __call__(self, prompt: str, seed: int = 42):
    '''    
    '''
    image = self.generate_image(prompt, seed)
    poses = self.detect_landmarks(image)
    if not poses:
      raise Exception("Failed to generate a pose!")

    annotated_image = self.annotate_landmarks(image, poses)
    pose_segments = []
    world_points = poses.pose_world_landmarks[0]
    for i, j in mps.pose.POSE_CONNECTIONS:
      p = world_points[i]
      q = world_points[j]
      pose_segments.append({i:(p.x,-p.y,-p.z), j:(q.x,-q.y,-q.z)})

    return annotated_image, pose_segments
    
  def as_service(self, port = SDPG_PORT):

    log.info("Running as a service on port {} (ctrl+c to terminate)...".format(port))
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('', port))
    try:
      while True:
        conn, addr = receiver.recvfrom(1024)
        data = json.loads(conn.decode())
        if 'prompt' in data:
          prompt = data['prompt']
          image = posegen.generate_image(prompt, data.get('seed',4093))
          poses = posegen.detect_landmarks(image)

          anno_img = posegen.annotate_landmarks(image, poses)
          anno_name = ''.join([c for c in prompt if c not in string.punctuation])
          anno_name = anno_name.replace(' ', '_')
          anno_path = os.path.abspath(f'SDPG_IMG/{anno_name}.jpg')
          if not os.path.exists(os.path.dirname(anno_path)):
            os.makedirs(os.path.dirname(anno_path))
          log.debug(f'Saving annotated image "{anno_path}"...')
          anno_img.save(anno_path, format='JPEG')
          
          if poses:
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            output = {'landmarks':[(p.x,-p.y,-p.z) for p in poses.pose_world_landmarks[0]]}
            output['image'] = anno_path
            sender.sendto(json.dumps(output).encode(), addr)
            log.debug("Pose gen data sent!")
    except KeyboardInterrupt:
      receiver.close()


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--cpu", action="store_true", help="Run inference on CPU")
  parser.add_argument("-b", "--base", type=str, default='sd15', choices=['sd15', 'sdxl'], help="Base SD model (sd15|sdxl)")
  parser.add_argument("-s", "--steps", type=int, choices=[2,4,8], default=8, help="Hyper-SD steps (2|4|8)")
  args = parser.parse_args()

  device = 'cpu'
  if not args.cpu:
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else device)
  base = 'SDXL' if args.base=='sdxl' else 'SD15'
  posegen = SimpleDiffusionPoseGen(args.steps, base, device)
  posegen.as_service()
