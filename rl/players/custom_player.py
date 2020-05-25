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

from rl.players import register_player
from rl.rl import ACTION, OBSERVATION
from models.custom_model import PPO


@register_player("default_player")
class DemoPlayer(object):
    def __init__(self, args, **kwargs):
        # self.model = setup_model(args)
        self.model = PPO
        self.agent = self.model.remote(args)
        self.args = args

    def step(self, obs: List[OBSERVATION]):
        obs = self._unwrap_observations(obs)
        actions_idx = self.agent.choose_actions.remote(obs)
        actions = ray.get(actions_idx)
        actions = self._wrap_actions(actions)
        return actions

    def _unwrap_observations(self, obs):
        return np.asarray([ob.observation for ob in obs])

    def _wrap_actions(self, actions) -> List[ACTION]:
        return [ACTION(a) for a in actions]
