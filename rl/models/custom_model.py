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

@file: ppo.py

@time: 20/05/2020 14:50 

@desc：       
               
'''
import tensorflow.compat.v1 as tf
import numpy as np
from typing import List

from .model import *
from rl.models import register_model


@register_model('ppo')
class PPO(RawModel):
    """
    This PPO version is adapted from Mofan Zhou, University of Technology Sydney.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--state_dim', type=int, default=3)
        parser.add_argument('--action_dim', type=int, default=1)
        parser.add_argument('--update_actor_steps', type=int, default=10)
        parser.add_argument('--update_critic_steps', type=int, default=10)
        parser.add_argument('--actor_lr', type=float, default=1e-4)
        parser.add_argument('--critic_lr', type=float, default=2e-4)
        parser.add_argument('--ppo_name', choices=['kl_gen', 'clip'], default='clip')
        parser.add_argument("--log_dir", type=str, default=None)

    def build_model(self, args, **kwargs):
        self.ppo_method = {
            'kl_pen': dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
            'clip': dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
        }.get(args.ppo_name)

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        )

        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.tfs = tf.placeholder(tf.float32, [None, args.state_dim], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(args.actor_lr).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, args.action_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
                surr = ratio * self.tfadv
            if args.ppo_name == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:  # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - self.ppo_method['epsilon'],
                                     1. + self.ppo_method['epsilon']) * self.tfadv))

        with tf.variable_scope('actor_train'):
            self.atrain_op = tf.train.AdamOptimizer(args.actor_lr).minimize(self.aloss)

        if args.log_dir:
            tf.summary.FileWriter(args.log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        print('Start optimizing ...')
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if self.args.ppo_name == 'kl_pen':
            for _ in range(self.args.update_actor_every):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: self.ppo_method['lam']})
                if kl > 4 * self.ppo_method['kl_target']:  # this in in google's paper
                    break
            if kl < self.ppo_method['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                self.ppo_method['lam'] /= 2
            elif kl > self.ppo_method['kl_target'] * 1.5:
                self.ppo_method['lam'] *= 2
                self.ppo_method['lam'] = np.clip(self.ppo_method['lam'], 1e-4,
                                                 10)  # sometimes explode, this clipping is my solution
        else:  # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in
             range(self.args.update_actor_steps)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(self.args.update_critic_steps)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.args.action_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.args.action_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_actions(self, s: np.ndarray) -> List[np.ndarray]:
        if s.ndim == 1:
            s = s[np.newaxis, :]
        bat_actions = self.sess.run(self.sample_op, {self.tfs: s})
        return [np.clip(a, -2, 2) for a in bat_actions]

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
