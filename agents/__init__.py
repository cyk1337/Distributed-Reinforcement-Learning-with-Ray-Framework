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

@file: __init__.py

@time: 20/05/2020 22:56 

@desc：       
               
'''
import os
import importlib

AGENT_REGISTRY = {}


def setup_agent(args):
    return AGENT_REGISTRY[args.agent_name]


def register_agent(name):
    """
    register your custom agent before class
    :param name:
    :return:
    """

    def register_agent_cls(cls):
        if name in AGENT_REGISTRY:
            raise ValueError("Player already registered!")
        AGENT_REGISTRY[name] = cls
        return cls

    return register_agent_cls


agents_dir = os.path.dirname(__file__)
for file in os.listdir(agents_dir):
    path = os.path.join(agents_dir, file)
    if not file.startswith('_') and not file.startswith(".") and (file.endswith('.py') or os.path.isdir(path)):
        agent_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'agents.{agent_name}')
