
from datetime import datetime
import functools
from IPython.display import HTML
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from etils import epath
from flax import struct
from matplotlib import pyplot as plt
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import cv2
from tqdm import tqdm

def save_video(frames, fps, output_file='output.mp4'):
  # Check if there are frames in the list
  if not frames:
    raise ValueError("The frame list is empty.")
  
  fps = int(fps)

  # Retrieve the shape of the first frame to initialize the video writer
  height, width, layers = frames[0].shape

  # Define the codec and create VideoWriter object
  # The fourcc code depends on the format desired, DIVX is used here as an example
  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
  out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

  # Write each frame to the video
  for frame in frames:
    out.write(frame)

  # Release everything when the job is finished
  out.release()
  print(f"Video saved as {output_file}")

def domain_randomize(sys, rng):
  """Randomizes the mjx.Model."""
  @jax.vmap
  def rand(rng):
    _, key = jax.random.split(rng, 2)
    # friction
    friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
    friction = sys.geom_friction.at[:, 0].set(friction)
    # actuator
    _, key = jax.random.split(key, 2)
    gain_range = (-5, 5)
    param = jax.random.uniform(
        key, (1,), minval=gain_range[0], maxval=gain_range[1]
    ) + sys.actuator_gainprm[:, 0]
    gain = sys.actuator_gainprm.at[:, 0].set(param)
    bias = sys.actuator_biasprm.at[:, 1].set(-param)
    return friction, gain, bias

  friction, gain, bias = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, sys)
  in_axes = in_axes.tree_replace({
      'geom_friction': 0,
      'actuator_gainprm': 0,
      'actuator_biasprm': 0,
  })

  sys = sys.tree_replace({
      'geom_friction': friction,
      'actuator_gainprm': gain,
      'actuator_biasprm': bias,
  })

  return sys, in_axes