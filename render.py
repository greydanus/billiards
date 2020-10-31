# Billiards Environment
# Sam Greydanus | 2020

import numpy as np
import contextlib

def render_balls(centers, r=.15, side=100):
  '''Renders the billiards environment.
  Note: 'centers' is a tensor of dims [time, balls, x_y_vx_vy_coords]'''
  has_time_dim = True
  if len(centers.shape) == 2:
    centers = centers[None,...]
    has_time_dim = False
  x = np.linspace(0, 1.0, side)[None, None, :]  # [time, balls, x]
  y = np.linspace(0, 1.0, side)[None, None, :]  # [time, balls, y]
  diff_x, diff_y = (x-centers[...,0:1])/r, (y-centers[...,1:2])/r
  diff_x, diff_y = diff_x[:,:, None, :], diff_y[:, :, :, None]
  frames = (diff_x**2+diff_y**2 < 1.0).astype(bool)  # mask is OFF in circle
  return frames if has_time_dim else frames.squeeze(0)

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        
def project_to_rgb(x, size=3, seed=1):
  '''Projects the last dimension to size(3).'''
  with temp_seed(seed): # make this section deterministically random
    P = np.random.rand(x.shape[-1], size)
    P /= np.linalg.norm(P,axis=-2, keepdims=True)
    P *= 255
  return np.einsum('...i,ij->...j', x, P).astype(np.int16)