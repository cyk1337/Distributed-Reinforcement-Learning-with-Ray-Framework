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

@file: env.py

@time: 20/05/2020 14:26 

@desc：       
               
'''
from abc import ABCMeta, abstractmethod


class RawEnvWrapper(metaclass=ABCMeta):
    def __init__(self, env, args, **kwargs):
        if not env:
            raise ValueError('Invalid environment provided!')
        self.env = env
        self.args = args
        self.done = None

    @staticmethod
    def add_args(parser):
        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        """ Env reset abstract method"""
        raise NotImplementedError

    @abstractmethod
    def step(self, trajectory, *args, **kwargs) -> tuple:
        """
        EnvWrapper step abstract method
        :param trajectory: trajectory object
        :param args: custom args
        :param kwargs: custom args, e.g., opponent_player
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _unwrap_action(self, action):
        """ Unwrap action to env.step args """
        raise NotImplementedError

    @abstractmethod
    def _wrap_observation(self, observation) -> tuple:
        """ Unwrap observation to OBSERVATION format
            :return (cur_ob, next_ob if done else None)
        """
        raise NotImplementedError
