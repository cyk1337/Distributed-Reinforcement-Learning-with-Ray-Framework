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

@file: __init__.py.py

@time: 20/05/2020 14:12 

@desc：       
               
'''

import os
import importlib

ENV_REGISTRY = {}


def setup_env(args):
    return ENV_REGISTRY[args.env_name]


def register_env(name):
    """
    register your custom env before class
    :param name:
    :return:
    """

    def register_env_cls(cls):
        if name in ENV_REGISTRY:
            raise ValueError("Model already registered!")
        ENV_REGISTRY[name] = cls
        return cls

    return register_env_cls


envs_dir = os.path.dirname(__file__)
for file in os.listdir(envs_dir):
    path = os.path.join(envs_dir, file)
    if not file.startswith('_') and not file.startswith(".") and (file.endswith('.py') or os.path.isdir(path)):
        env_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'rl.envs.{env_name}')
