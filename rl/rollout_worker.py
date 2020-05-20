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

@file: rollout_worker.py

@time: 20/05/2020 14:12 

@desc：       
               
'''
from typing import List
from abc import ABCMeta, abstractmethod
from rl.rl import ACTION, TRAJECTORY
from rl import envs


class RawRolloutWorkerPool(metaclass=ABCMeta):
    def __init__(self, learner, args, **kwargs):
        self.env = envs.setup_env(args)
        self.learner = learner
        self.player = learner.player
        self.num_workers = args.num_workers
        self.trajectory = [[] for _ in range(args.num_workers)]
        self.traj_len = args.traj_len
        self.worker_pool = [self.env for _ in range(args.num_workers)]
        self.s_ = None
        self.GAMMA = args.GAMMA

    def start(self):
        init_states = [env.reset() for env in self.worker_pool]
        self.s_ = init_states
        return init_states

    def get_player(self):
        return self.player

    @abstractmethod
    def run(self, *args, **kwargs):
        """ Start batch envs"""
        raise NotImplementedError

    def __repr__(self):
        return 'Base rollout worker'


class DemoRolloutWorkerPool(RawRolloutWorkerPool):
    def __init__(self, env, args):
        super().__init__(env, args)

    def run(self, actions, *args, **kwargs) -> List[ACTION]:
        """ Start bat envs"""
        bat_obs = []
        for i, (env, action) in enumerate(zip(self.worker_pool, actions)):
            ob = env.step(action)
            # self.trajectory[i].append(
            #     TRAJECTORY(self.s_[i], ob.reward, ob.done, ob.action, None, None)
            # )
            # self.s_[i] = ob.observation
            bat_obs.append(ob)
            #
            # if ob.done or len(self.trajectory[i]) > self.traj_len:
            #     v_s_ = self.player.agent.get_v(ob.observation)
            #     for traj in self.trajectory[i][::-1]:
            #         traj.R = traj.reward + self.GAMMA * v_s_
            #     self.learner.send_trajectory(self.trajectory[i])
            #     self.trajectory[i] = []

        return bat_obs


def __repr__(self):
    return f'Demo rollout "{self.num_workers}" workers  on [{self.env}]'
