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

@file: learner.py

@time: 20/05/2020 19:02 

@desc：       
               
'''
import tensorflow.compat.v1 as tf
from rl.learners import register_learner
from rl import players


@register_learner('default_learner')
class Learner(object):
    def __init__(self, args):
        self.player = players.setup_player(args)
        self.trajectories = []
        self.optimizer = None
        self.batch_size = args.batch_size

    @staticmethod
    def add_args(parser):
        parser.add_argument('--batch_size', type=int, default=10)

    def get_parameters(self):
        return self.player.agent.get_weights()

    def sed_trajectories(self, trajectory):
        self.trajectories.append(trajectory)

    def update_parameters(self):
        trajectories = self.trajectories[:self.batch_size]
        self.trajectories = self.trajectories[self.batch_size:]
        loss = None
