# Billiards Environment
# Sam Greydanus | 2020

import numpy as np
from .utils import ObjectView, to_pickle, from_pickle
from .simulate import Billiards


def get_dataset_args(as_dict=False):
    arg_dict = {'num_samples': 10000,
                'train_split': 0.9,
                'time_steps': 45,
                'num_balls': 2,
                'r': 1e-1,
                'dt': 1e-2,
                'seed': 0,
                'make_1d': False,
                'verbose': True,
                'side': 32,  # side lenth, in pixels
                'use_pixels': False}
    return arg_dict if as_dict else ObjectView(arg_dict)


def make_trajectory(env, args):
  obs, coords, actions = [], [], []
  next_action = None
  for i in range(args.time_steps):
    o, r, d, info = env.step(next_action)
    next_action = 1.2 * (2*np.random.rand((2))-1) if i==3 else np.zeros((2))
    obs.append(o) ; coords.append(info['position']) ; actions.append(next_action.copy())
  return np.stack(obs), np.stack(coords), np.stack(actions)

def make_dataset(args, **kwargs):
  if args.use_pixels and args.verbose:
    print('When Sam profiled this code, it took 0.15 sec/trajectory.')
    print('\t-> Expect it to take ~25 mins to generate 10k samples.')
    
  np.random.seed(args.seed)
  env = Billiards(args, use_pixels=args.use_pixels)
  xs, cs = [], []  # xs, which may be pixels, and cs, which are always coordinates, acts=actions
  for i in range(args.num_samples):
    x, c, a = make_trajectory(env, args)
    c = np.concatenate([c,a], axis=-1)
    if not args.use_pixels:
        x = c  # if making a coord dataset, include action info in observation
    xs.append(x) ; cs.append(c) ; env.reset()
    if args.verbose and (i+1)%10==0:
      print('\rdataset {:.2f}% built'.format((i+1)/args.num_samples * 100), end='', flush=True)

  xs, cs = [np.stack(v).swapaxes(0,1) for v in [xs, cs]]
  split_ix = int(args.num_samples*args.train_split) # train / test split
  dataset = {'x': xs[:, :split_ix], 'x_test': xs[:, split_ix:],
            'dt': args.dt, 'r': args.r, 'num_balls': args.num_balls}
  if args.use_pixels:
    dataset['c'] = cs[:, :split_ix]
    dataset['c_test'] = cs[:, split_ix:]
  return dataset

# we'll cache the dataset so that it doesn't have to be rebuild every time
def load_dataset(args, path=None, regenerate=False, **kwargs):
    path = './billiards.pkl' if path is None else path
    try:
      if regenerate:
          raise ValueError("Regenerating dataset") # yes this is hacky
      dataset = from_pickle(path)
      if args.verbose:
          print("Successfully loaded data from {}".format(path))
    except:
      if args.verbose:
          print("Did or could not load data from {}. Rebuilding dataset...".format(path))
      dataset = make_dataset(args, **kwargs)
      to_pickle(dataset, path)
    return dataset