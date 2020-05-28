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

@file: rl.py

@time: 20/05/2020 14:13 

@desc：       
               
'''

from collections import namedtuple

OBSERVATION_FIELDS = [
    'observation',
    'reward',
    'done',
]

ACTION_FIELDS = [
    'action',
]

TRAJECTORY_FIELDS = [
    'observation',
    'reward',
    'done',
    'action',
    'behavior_logits',
    'R',
]

OBSERVATION = namedtuple('Observation', OBSERVATION_FIELDS)
ACTION = namedtuple('Action', ACTION_FIELDS)
TRAJECTORY = namedtuple('Trajectory', TRAJECTORY_FIELDS)
