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

@file: custom_options.py

@time: 21/05/2020 15:51 

@desc：       
               
'''


def get_custom_args(parser):
    """ add custom arguments here """
    # rl training
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ep_max', type=int, default=10000)
    parser.add_argument('--traj_len', type=int, default=10, help='Trajectory length')
    parser.add_argument('--GAMMA', type=float, default=.9, help='Siscounting factor')

    parser.add_argument('--gpu', type=str, default=None)
    # parser.add_argument('--resume_last', action='store_true')
    # parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--debug', action='store_true')
