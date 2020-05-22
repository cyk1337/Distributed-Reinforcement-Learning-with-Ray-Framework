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

@file: player.py

@time: 20/05/2020 20:47 

@desc：       
               
'''
from rl.utils import *
from rl.models import setup_model
from rl.players import register_player
from rl.rl import ACTION


class RawPlayer(metaclass=ABCMeta):
    def __init__(self, args, **kwargs):
        self.agent = setup_model(args)
        self.args = args

    @abstractmethod
    def step(self, obs):
        raise NotImplementedError

    @abstractmethod
    def _wrap_actions(self, actions):
        raise NotImplementedError

    @abstractmethod
    def _unwrap_observations(self, obs):
        raise NotImplementedError


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
