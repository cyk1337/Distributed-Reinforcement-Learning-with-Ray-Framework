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
import numpy as np
from typing import List

from rl.players import register_player
from rl.players.player import RawPlayer
from rl.rl import ACTION, OBSERVATION


@register_player("demo_random_player")
class DemoRandomPlayer(RawPlayer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        import gym
        env = gym.make('Pendulum-v0').unwrapped
        self.action_space = env.action_space

    def step(self, obs):
        actions = []
        for _ in range(len(obs)):
            a = self.action_space.sample()
            actions.append(ACTION(a))
        return actions

    def _wrap_actions(self, actions):
        pass


@register_player("default_player")
class DemoPlayer(RawPlayer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def step(self, obs: List[OBSERVATION]):
        obs = self._unwrap_observations(obs)
        actions = self.agent.choose_actions(obs)
        actions = self._wrap_actions(actions)
        return actions

    def _unwrap_observations(self, obs):
        return np.asarray([ob.observation for ob in obs])

    def _wrap_actions(self, actions) -> List[ACTION]:
        return [ACTION(a) for a in actions]
