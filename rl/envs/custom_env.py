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

@file: custom_env.py

@time: 20/05/2020 23:08 

@desc：       
               
'''
from rl.envs.env import RawEnvWrapper
from rl.rl import OBSERVATION
from rl.envs import register_env


@register_env('gym_env')
class DemoEnvWrapper(RawEnvWrapper):
    def __init__(self, args, **kwargs):
        import gym
        env = gym.make('Pendulum-v0').unwrapped
        super().__init__(env, args, **kwargs)

    def reset(self, *args, **kwargs):
        # custom operation
        ob = self.env.reset()
        init_ob = OBSERVATION(ob, None, None)
        return init_ob

    def step(self, action, *args, **kwargs):
        action = self._unwrap_action(action)
        observation = self.env.step(action)
        return self._wrap_observation(observation)

    def _unwrap_action(self, action):
        return action.action

    def _wrap_observation(self, observation):
        ob, reward, done, _ = observation
        observation = OBSERVATION(ob, reward, done)
        return observation

    def __repr__(self):
        return 'Demo environment'
