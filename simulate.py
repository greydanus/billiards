# Billiards Environment
# Sam Greydanus | 2020

import numpy as np
from skimage.transform import resize
from .render import render_masks, project_to_rgb

############# Basic Physics Simulation ############# 

def rotation_matrix(theta):  # contruct a rotation matrix
  return np.asarray([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

def rotate(x, theta):  # rotate vector x by angle theta
  R = rotation_matrix(theta)
  return (x.reshape(1,-1) @ R)[0]

def angle_between(v0, v1):  # the angle between two vectors
  return np.math.atan2(np.linalg.det([v0,v1]), np.dot(v0,v1))

def reflect(x, axis):  # reflect vector x about some other vector 'axis'
  new_xs = np.zeros_like(x)
  for i in range(x.shape[0]):
    theta = angle_between(x[i], axis[i])
    if np.abs(theta) > np.pi/2:
      theta = theta + np.pi
    new_xs[i] = rotate(-x[i], 2 * -theta)
  return new_xs
  
def collide_walls(xs, vs, r, dt):
  mask_low = np.where(xs < r)  # coordinates that are too low
  mask_high = np.where(xs > 1-r)  # coordinates that are too high
  vs[mask_low] *= -1  # rebound
  vs[mask_high] *= -1
  xs[mask_low] = 2*r - xs[mask_low]  # account for overshooting the wall
  xs[mask_high] = (1-r) - (xs[mask_high] - (1-r))  # easier to understand
  return xs, vs

def find_colliding_balls(xs, r):
  dist_matrix = ((xs[:,0:1] - xs[:,0:1].T)**2 + (xs[:,1:2] - xs[:,1:2].T)**2)**.5
  dist_matrix[np.tril_indices(xs.shape[0])] = np.inf  # we only care about upper triangle
  body1_mask, body2_mask = np.where(dist_matrix < 2*r)  # select indices of colliding balls
  return body1_mask, body2_mask

def collide_balls(new_xs, vs, r, dt):
  body1_mask, body2_mask = find_colliding_balls(new_xs, r)
  
  # if at least one pair of balls are colliding
  if len(body1_mask) > 0:
    radii_diff = new_xs[body2_mask] - new_xs[body1_mask]  # diff. between radii

    prev_xs = new_xs - vs * dt  # step backward in time
    prev_radii_diff = prev_xs[body2_mask] - prev_xs[body1_mask]

    # if the pair of balls are getting closer to one another
    if np.sum(radii_diff**2) < np.sum(prev_radii_diff**2):
      vs_body1, vs_body2 = vs[body1_mask], vs[body2_mask]  # select the two velocities
      v_com = (vs_body1 + vs_body2) / 2   # find the velocity of the center of masses (assume m1=m2)
      vrel_body1 = vs_body1 - v_com  # we care about relative velocities of the ball

      reflected_vrel_body1 = reflect(vrel_body1, radii_diff)
      vs[body1_mask] = reflected_vrel_body1 + v_com  # rotate velocities (assumes m1=m2)
      vs[body2_mask] = -reflected_vrel_body1 + v_com # symmetry of a perfect collision

  return new_xs, vs

def init_balls(r, num_balls=3, make_1d=False, normalize_v=False):
  x0 = np.random.rand(num_balls, 2) * (1-2*r) + r  # balls go anywhere in box
  v0 = (.75 * np.random.randn(*x0.shape)).clip(-1.2, 1.2)
  if make_1d:
    x0[:,0] = 0.5 ; v0[:,0] = 0  # center and set horizontal velocity to 0
  if normalize_v:
    v0 /= np.linalg.norm(v0, axis=1, keepdims=True)  # velocities start out normalized
  mask, _ = find_colliding_balls(x0, r)  # recursively re-init if any balls overlap
  return init_balls(r, num_balls, make_1d) if len(mask) > 0 else np.concatenate((x0, v0),axis=-1)

def simulate_balls(r=8e-2, dt=2e-2, num_steps=50, num_balls=2, init_state=None, make_1d=False,
                   normalize_v=False, verbose=False):
  start_state = init_balls(r, num_balls, make_1d, normalize_v) if init_state is None else init_state
  x0, v0 = start_state[:,:2], start_state[:,2:]
  # x0, v0 = np.flip(x0, axis=0), np.flip(v0, axis=0)  # debugging: simulation should be invariant to this

  curr_x, curr_v = x0, v0
  xs, vs = [x0.copy()], [v0.copy()]
  if verbose: print('initial energy: ', (curr_v**2).sum())
  for i in range(num_steps-1):
    new_xs = xs[-1] + curr_v * dt
    new_xs, curr_v = collide_walls(new_xs, curr_v, r, dt)
    new_xs, curr_v = collide_balls(new_xs, curr_v, r, dt)
    xs.append(new_xs.copy())
    vs.append(curr_v.copy())
  if verbose: print('final energy: ', (curr_v**2).sum())
  return np.concatenate([np.stack(xs), np.stack(vs)], axis=-1)


############# RL Environment Wrapper ############# 

class Billiards:
  def __init__(self, args, use_pixels=False):
    assert not args.make_1d, "We only support 2D sims"
    self.make_1d = False
    self.r = args.r
    self.num_balls = args.num_balls
    self.dt = args.dt
    self.seed = args.seed
    self.args = args
    self.use_pixels = use_pixels
    self.side = args.side
    self.reset()
    
  def reset(self):
    self.state = state = init_balls(self.r, self.num_balls, self.make_1d, normalize_v=False)
    # state has shape [balls, xyvxvy]
    self.x, self.v = state[:,:2], state[:,2:]

  def step(self, action=None, num_steps=5, tau=1):
    if action is None:
        action = np.zeros((2)) # force applied to second ball
    assert action.shape[0] == 2
    action = action.clip(-tau, tau)  # maximum force that can be applied
    self.state[1,2:] += action  # given v' = a*t + v & F=ma, we set m=t=1 and get v' = F + v
    
    state = simulate_balls(self.r, self.dt, num_steps, self.num_balls, self.state,
                           self.make_1d, normalize_v=False, verbose=False)[-1]
    # state has shape [balls, xyvxvy]
    self.state = state
    self.x, self.v = state[:,:2], state[:,2:]
    
    done = (self.x[0,0] > 0.8) and (self.x[0,1] < 0.2) # ball 0 is in upper right corner
    reward = 1. if done else 0.
    info = {'position': self.x.flatten(), 'velocity': self.v}
    
    if self.use_pixels:
      masks = render_masks(state, r=self.r, side=3*self.side).transpose(1,2,0) # masks has shape [x,y,num_balls]
      obs = project_to_rgb(masks)
      obs = resize(obs.astype(float), (self.side, self.side, 3)).astype(np.int16)
      if done:
        obs[:1] = obs[-1:] = obs[:,:1] = obs[:,-1:] = 100  # border color changes when reward is received
    else:
      obs = self.x.flatten()
    # obs has shape [x, y, rgb] if use_pixels, otherwise shape [balls * xyvxvy]
    return obs, reward, done, info