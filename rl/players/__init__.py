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

PLAYER_REGISTRY = {}
PLAYER_CLASS_NAMES = set()


def setup_player(args, **kwargs):
    return PLAYER_REGISTRY[args.player_name](args, **kwargs)


def register_player(name):
    """
    register your custom player before class
    :param name:
    :return:
    """

    def register_player_cls(cls):
        if name in PLAYER_REGISTRY:
            raise ValueError("Player already registered!")
        PLAYER_REGISTRY[name] = cls
        PLAYER_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_player_cls


players_dir = os.path.dirname(__file__)
for file in os.listdir(players_dir):
    path = os.path.join(players_dir, file)
    if not file.startswith('_') and not file.startswith(".") and (file.endswith('.py') or os.path.isdir(path)):
        player_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'rl.players.{player_name}')
