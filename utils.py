# Billiards Environment
# Sam Greydanus | 2020

import pickle

def states_to_xy(states):
  '''A helper function for render_masks'''
  return states[...,:4].reshape(*states.shape[:-1],2,2)

class ObjectView(object):  # make a dictionary look like an object
  def __init__(self, d): self.__dict__ = d


def to_pickle(thing, path):  # save something
  with open(path, 'wb') as handle:
      pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path):  # load something
  thing = None
  with open(path, 'rb') as handle:
      thing = pickle.load(handle)
  return thing