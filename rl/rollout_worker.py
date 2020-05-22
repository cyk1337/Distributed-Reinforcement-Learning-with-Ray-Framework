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
import ray
from typing import List
import numpy as np
from abc import ABCMeta, abstractmethod

from rl.rl import ACTION, OBSERVATION, TRAJECTORY
from rl import envs


class RawRolloutWorkerPool(metaclass=ABCMeta):
    def __init__(self, learner, args, **kwargs):
        self.learner = learner
        self.player = learner.get_player()
        self.num_workers = args.num_workers
        self.trajectory = [[] for _ in range(args.num_workers)]
        self.env = envs.setup_env(args)
        self.worker_pool = [self.env.remote(args) for _ in range(args.num_workers)]
        self.traj_len = args.traj_len
        self.s_ = None
        self.GAMMA = args.GAMMA

    @abstractmethod
    def start(self) -> List[OBSERVATION]:
        raise NotImplementedError

    def get_player(self):
        return self.player

    @abstractmethod
    def run(self, *args, **kwargs):
        """ Start batch envs"""
        raise NotImplementedError

    def __repr__(self):
        return 'Base rollout worker'


class DistRolloutWorkerPool(RawRolloutWorkerPool):
    def __init__(self, env, args):
        super().__init__(env, args)

    def start(self) -> List[OBSERVATION]:
        init_states = ray.get([env.reset.remote() for env in self.worker_pool])
        self.s_ = [s.observation for s in init_states]
        return init_states

    def run(self, actions, *args, **kwargs) -> List[ACTION]:
        """ Start bat envs"""
        print('Start rollouts workers ...')
        bat_obs = []
        for i, obj_id in enumerate([env.step.remote(action) for (env, action) in zip(self.worker_pool, actions)]):
            ob = ray.get(obj_id)
            self.trajectory[i].append(
                TRAJECTORY(self.s_[i], ob.reward, ob.done, actions[i].action, None, None)
            )

            if ob.done or len(self.trajectory[i]) >= self.traj_len:
                v_s_ = self.player.agent.get_v(ob.observation)
                discounted_r = []
                for traj in self.trajectory[i][::-1]:
                    v_s_ = traj.reward + self.GAMMA * v_s_
                    discounted_r.append(v_s_)
                    discounted_r.reverse()
                self.learner.send_trajectory(self.trajectory[i], np.array(discounted_r)[:, np.newaxis])
                self.trajectory[i] = []

            if ob.done:
                ob = ray.get(envs[i].reset.remote())
            self.s_[i] = ob.observation
            bat_obs.append(ob)

        return bat_obs

    def __repr__(self):
        return f'Demo rollout "{self.num_workers}" workers'


class DemoRolloutWorkerPool(RawRolloutWorkerPool):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.worker_pool = [self.env for _ in range(args.num_workers)]

    def start(self) -> List[OBSERVATION]:
        init_states = [env.reset() for env in self.worker_pool]
        self.s_ = [s.observation for s in init_states]
        return init_states

    def run(self, actions, *args, **kwargs) -> List[ACTION]:
        """ Start bat envs"""
        print('Start rollouts workers ...')
        bat_obs = []
        for i, (env, action) in enumerate(zip(self.worker_pool, actions)):
            ob = env.step(action)
            self.trajectory[i].append(
                TRAJECTORY(self.s_[i], ob.reward, ob.done, action.action, None, None)
            )

            if ob.done or len(self.trajectory[i]) >= self.traj_len:
                v_s_ = self.player.agent.get_v(ob.observation)
                discounted_r = []
                for traj in self.trajectory[i][::-1]:
                    v_s_ = traj.reward + self.GAMMA * v_s_
                    discounted_r.append(v_s_)
                    discounted_r.reverse()
                self.learner.send_trajectory(self.trajectory[i], np.array(discounted_r)[:, np.newaxis])
                self.trajectory[i] = []

            if ob.done:
                ob = env.reset()
            self.s_[i] = ob.observation
            bat_obs.append(ob)

        return bat_obs

    def __repr__(self):
        return f'Demo rollout "{self.num_workers}" workers  on [{self.env}]'
