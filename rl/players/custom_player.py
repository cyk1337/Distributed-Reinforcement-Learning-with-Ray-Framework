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
from rl.players import register_player
from rl.players.player import RawPlayer
from rl.rl import ACTION


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


@register_player("default_player")
class DemoPlayer(RawPlayer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def step(self, obs):
        pass
