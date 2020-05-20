#!/usr/bin/env python

# -*- encoding: utf-8

'''
    _____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
    \__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
     /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
     \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
     / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
     \/               \/        \/               \/          \/       \/         \/     
 
 ==========================================================================================

@author: Yekun Chai

@license: School of Informatics, Edinburgh

@contact: chaiyekun@gmail.com

@file: rl.py

@time: 20/05/2020 14:13 

@desc：       
               
'''

from collections import namedtuple
from random import randint

OBSERVATION_FIELDS = [
    'observation',
    'reward',
    'done',
]

ACTION_FIELDS = [
    'action',
]

TRAJECTORY_FIELDS = [
    'observation',
    'reward',
    'done',
    'action',
    'behavior_logits',
    'R',
]

OBSERVATION = namedtuple('Observation', OBSERVATION_FIELDS)
ACTION = namedtuple('Action', ACTION_FIELDS)
TRAJECTORY = namedtuple('Trajectory', TRAJECTORY_FIELDS)


class Trajectory(object):
    def __init__(self, max_size=None):
        self._storage = []
        self._max_size = max_size
        self._next_idx = 0

    def add(self, traj):
        if self._next_idx >= len(self._storage):
            self._storage.append(traj)
        else:
            self._storage[self._next_idx] = traj
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, bsz: int):
        return []

    def __len__(self):
        return len(self._storage)
