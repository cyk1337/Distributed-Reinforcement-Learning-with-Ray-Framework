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

@file: custom_player.py

@time: 20/05/2020 23:07 

@desc：       
               
'''
from typing import List
import numpy as np
import ray

from agents import register_agent
from rl import ACTION, OBSERVATION


@register_agent("default_agent")
class DefaultAgent(object):
    def __init__(self, args, model):
        self.model = model
        self.trajectories = []
        self.args = args
        self.batch_size = args.batch_size

    @staticmethod
    def add_args(parser):
        """ add custom arguments here """
        parser.add_argument('--batch_size', type=int, default=5)

    def set_weights(self, weights):
        f = self.model.set_weights
        f.remote(*weights) if hasattr(f, 'remote') else f(*weights)

    @ray.method(num_return_vals=2)
    def get_weights(self):
        f = self.model.get_weights
        a, c = f.remote() if hasattr(f, 'remote') else f()
        return a, c

    def step(self, obs: List[OBSERVATION]):
        obs = self._unwrap_observations(obs)
        actions_idx = self.model.choose_actions.remote(obs)
        actions = ray.get(actions_idx)
        actions = self._wrap_actions(actions)
        return actions

    def _unwrap_observations(self, obs):
        return np.asarray([ob.observation for ob in obs])

    def _wrap_actions(self, actions) -> List[ACTION]:
        return [ACTION(a) for a in actions]

    def send_trajectory(self, trajectory, Rs=None):
        self.trajectories.append([trajectory, Rs])

    def fetch_grads(self, weights):
        # while True:
        if len(self.trajectories) >= self.batch_size:
            grads = self.compute_gradients(weights)
            print(f'start learning \n {"*"*80}')
            return grads

    def compute_gradients(self, weights):
        trajectories = self.trajectories[:self.batch_size]
        self.trajectories = self.trajectories[self.batch_size:]
        # prepare trajectories
        bat_obs, bat_actions, bat_Rs = self.pack_trajectories(trajectories)
        # get gradients
        grads = self.model.step.remote(bat_obs, bat_actions, bat_Rs, *weights)
        return grads

    def pack_trajectories(self, trajectories):
        bat_obs = []
        bat_actions = []
        bat_Rs = []
        for item in trajectories:
            for traj, v in zip(*item):
                bat_obs.append(traj.observation)
                bat_actions.append(traj.action)
                bat_Rs.append(v)
        return bat_obs, bat_actions, bat_Rs
