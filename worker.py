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

from rl import ACTION, OBSERVATION, TRAJECTORY
import envs


class RawRolloutWorkerPool(metaclass=ABCMeta):
    def __init__(self, args, agent, model, **kwargs):
        self.agent = agent
        self.model = model
        self.num_envs_per_worker = args.num_envs_per_worker
        self.trajectory = [[] for _ in range(args.num_envs_per_worker)]
        self.env = envs.setup_env(args)
        self.worker_pool = [self.env.remote(args) for _ in range(args.num_envs_per_worker)]
        self.traj_len = args.traj_len
        self.s_ = None
        self.GAMMA = args.GAMMA

    @abstractmethod
    def start(self) -> List[OBSERVATION]:
        raise NotImplementedError

    def __repr__(self):
        return 'Base rollout worker'


@ray.remote
class DefaultWorker(RawRolloutWorkerPool):
    def __init__(self, args, agent, model):
        super().__init__(args, agent, model)

    def start(self) -> List[OBSERVATION]:
        init_states = ray.get([env.reset.remote() for env in self.worker_pool])
        self.s_ = [s.observation for s in init_states]
        return init_states

    def step(self, actions, *args, **kwargs) -> List[ACTION]:
        """ Start bat envs"""
        print('Step workers ...')
        bat_obs = []
        remaining_ids = [env.step.remote(action) for (env, action) in zip(self.worker_pool, actions)]
        ready_ids, remaining_ids = ray.wait(remaining_ids)
        for i, obj_id in enumerate(ready_ids):
            ob = ray.get(obj_id)
            self.trajectory[i].append(
                TRAJECTORY(self.s_[i], ob.reward, ob.done, actions[i].action, None, None)
            )

            if ob.done or len(self.trajectory[i]) >= self.traj_len:
                v_s_ = ray.get(self.model.get_v.remote(ob.observation))
                discounted_r = []
                for traj in self.trajectory[i][::-1]:
                    v_s_ = traj.reward + self.GAMMA * v_s_
                    discounted_r.append(v_s_)
                    discounted_r.reverse()
                self.agent.send_trajectory.remote(self.trajectory[i], np.array(discounted_r)[:, np.newaxis])
                self.trajectory[i] = []

            if ob.done:
                ob = ray.get(envs[i].reset.remote())
            self.s_[i] = ob.observation
            bat_obs.append(ob)
        return bat_obs

    def __repr__(self):
        return f'Demo rollout "{self.num_envs_per_worker}" envs'
