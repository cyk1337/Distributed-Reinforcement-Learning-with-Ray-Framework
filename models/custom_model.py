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
import ray
import tensorflow.compat.v1 as tf
import numpy as np
from typing import List
from ray.experimental.tf_utils import TensorFlowVariables

from models import register_model


@register_model('ppo')
class PPO(object):
    """
    This PPO version is adapted from Mofan Zhou, University of Technology Sydney.
    """

    def __init__(self, args):
        self.args = args
        self.build_model(args)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--state_dim', type=int, default=3)
        parser.add_argument('--action_dim', type=int, default=1)
        # parser.add_argument('--update_actor_steps', type=int, default=10)  # unused in ray
        # parser.add_argument('--update_critic_steps', type=int, default=10)  # unused in ray
        parser.add_argument('--actor_lr', type=float, default=1e-4)
        parser.add_argument('--critic_lr', type=float, default=2e-4)
        parser.add_argument('--clip_norm', type=float, default=5)
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
        self.critic_opt = tf.train.AdamOptimizer(args.critic_lr)
        self.c_grads = self.critic_opt.compute_gradients(self.closs)
        self.ctrain_op = self.critic_opt.apply_gradients(self.c_grads)
        self.c_vars = TensorFlowVariables(self.closs, self.sess)

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
            self.actor_opt = tf.train.AdamOptimizer(args.actor_lr)
            self.a_grads = self.actor_opt.compute_gradients(self.aloss)
            self.a_grads = [(t, v) for t, v in self.a_grads if t is not None]
            self.atrain_op = self.actor_opt.apply_gradients(self.a_grads)
            self.a_vars = TensorFlowVariables(self.aloss, self.sess)

        if args.log_dir:
            tf.summary.FileWriter(args.log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    @ray.method(num_return_vals=2)
    def step(self, s, a, r, a_weights, c_weights):
        self.set_weights(a_weights, c_weights)

        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

        # update actor
        a_grads = self.sess.run([grad[0] for grad in self.a_grads], {self.tfs: s, self.tfa: a, self.tfadv: adv})
        # update critic
        c_grads = self.sess.run([grad[0] for grad in self.c_grads], {self.tfs: s, self.tfdc_r: r})
        return a_grads, c_grads

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

    @ray.method(num_return_vals=2)
    def get_weights(self):
        return self.a_vars.get_weights(), self.c_vars.get_weights()

    def set_weights(self, a_weights, c_weights):
        self.a_vars.set_weights(a_weights)
        self.c_vars.set_weights(c_weights)
