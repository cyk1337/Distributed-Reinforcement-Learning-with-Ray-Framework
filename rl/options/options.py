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

@file: options.py

@time: 20/05/2020 21:12 

@desc：       
               
'''

import argparse
from rl.options.custom_options import get_custom_args


def get_training_parser(default_env='gym_env',
                        default_model='ppo',
                        default_player='default_player',
                        default_learner='default_learner'):
    parser = get_parser('Dist.ray.train')
    add_env_args(parser, default_env)
    add_player_args(parser, default_player)
    add_learner_args(parser, default_learner)
    add_model_args(parser, default_model)
    add_save_args(parser)
    return parser


def get_parser(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--exp_name', type=str, default='rl.distributed.ray.demo')
    return parser


def add_save_args(parser):
    group = parser.add_argument_group('Model save configuration')
    group.add_argument('--save_model_path', default='saved_models', type=str)
    group.add_argument('--logs_folder', type=str, default='tensorboard_logs')


def add_env_args(parser, default_env):
    group = parser.add_argument_group('Env configuration')
    from rl.envs import ENV_REGISTRY
    group.add_argument('--env_name', choices=[ENV_REGISTRY.keys()], default=default_env)
    args, _ = parser.parse_known_args()
    cls = ENV_REGISTRY[args.env_name]
    if hasattr(cls, "add_args"):
        cls.add_args(parser)
    return group


def add_player_args(parser, default_player):
    group = parser.add_argument_group('Player configuration')
    from rl.players import PLAYER_REGISTRY
    group.add_argument('--player_name', choices=[PLAYER_REGISTRY.keys()], default=default_player)
    args, _ = parser.parse_known_args()
    cls = PLAYER_REGISTRY[args.player_name]
    if hasattr(cls, "add_args"):
        cls.add_args(parser)
    return group


def add_learner_args(parser, default_learner):
    group = parser.add_argument_group('Player configuration')
    from rl.learners import LEARNER_REGISTRY
    group.add_argument('--learner_name', choices=[LEARNER_REGISTRY.keys()], default=default_learner)
    args, _ = parser.parse_known_args()
    cls = LEARNER_REGISTRY[args.learner_name]
    if hasattr(cls, "add_args"):
        cls.add_args(parser)
    return group


def add_model_args(parser, default_model):
    group = parser.add_argument_group('Model configuration')
    from rl.models import MODEL_REGISTRY
    group.add_argument('--model_name', choices=[MODEL_REGISTRY.keys()], default=default_model)
    args, _ = parser.parse_known_args()
    cls = MODEL_REGISTRY[args.model_name]
    if hasattr(cls, "add_args"):
        cls.add_args(parser)
    return group


def parse_custom_args(parser):
    get_custom_args(parser)
    args = parser.parse_args()
    return args