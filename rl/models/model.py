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

@file: model.py

@time: 20/05/2020 20:53 

@desc：       
               
'''
from rl.utils import *


class RawModel(metaclass=ABCMeta):
    def __init__(self, args, **kwargs):
        self.args = args
        self.build_model(args, **kwargs)

    @staticmethod
    def add_args(parser):
        raise NotImplementedError

    @abstractmethod
    def build_model(self, args, **kwargs):
        raise NotImplementedError
