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
from rl.learners import register_learner
from rl import players


@register_learner('default_learner')
class Learner(object):
    def __init__(self, args):
        self.player = players.setup_player(args)
        self.trajectories = []
        self.batch_size = args.batch_size

    @staticmethod
    def add_args(parser):
        """ add custom arguments here """
        parser.add_argument('--batch_size', type=int, default=5)

    def send_trajectory(self, trajectory, Rs=None):
        self.trajectories.append([trajectory, Rs])

    def run(self):
        # while True:
        if len(self.trajectories) >= self.batch_size:
            self.update_parameters()

    def update_parameters(self):
        trajectories = self.trajectories[:self.batch_size]
        self.trajectories = self.trajectories[self.batch_size:]
        bat_obs, bat_actions, bat_Rs = self.pack_trajectories(trajectories)
        self.player.agent.update(bat_obs, bat_actions, bat_Rs)

    def pack_trajectories(self, trajectories):
        # todo: pack trajectories
        bat_obs = []
        bat_actions = []
        bat_Rs = []
        for item in trajectories:
            for traj, v in zip(*item):
                bat_obs.append(traj.observation)
                bat_actions.append(traj.action)
                bat_Rs.append(v)
        return bat_obs, bat_actions, bat_Rs
