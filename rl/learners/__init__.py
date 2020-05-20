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

@time: 20/05/2020 23:54 

@desc：       
               
'''
import os
import importlib

LEARNER_REGISTRY = {}
LEARNER_CLASS_NAMES = set()


def setup_learner(args, **kwargs):
    return LEARNER_REGISTRY[args.learner_name](args, **kwargs)


def register_learner(name):
    """
    register your custom player before class
    :param name:
    :return:
    """

    def register_learner_cls(cls):
        if name in LEARNER_REGISTRY:
            raise ValueError("Learner already registered!")
        LEARNER_REGISTRY[name] = cls
        LEARNER_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_learner_cls


learners_dir = os.path.dirname(__file__)
for file in os.listdir(learners_dir):
    path = os.path.join(learners_dir, file)
    if not file.startswith('_') and not file.startswith(".") and (file.endswith('.py') or os.path.isdir(path)):
        learner_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'rl.learners.{learner_name}')
