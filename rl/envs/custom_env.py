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
        return self.env.reset()

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
        # if done:
        #     new_ob, new_reward, new_done, _ = self.env.reset()
        #     new_observation = OBSERVATION(new_ob, reward, done)
        # else:
        #     new_observation = None
        # return observation, new_observation

    def __repr__(self):
        return 'Demo environment'
