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
import os


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

    # mdoel
    parser.add_argument('--state_dim', type=int, default=3)
    parser.add_argument('--action_dim', type=int, default=1)
    parser.add_argument('--update_actor_steps', type=int, default=10)
    parser.add_argument('--update_critic_steps', type=int, default=10)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=2e-4)
    parser.add_argument('--ppo_name', choices=['kl_gen', 'clip'], default='clip')
    parser.add_argument("--log_dir", type=str, default=None)


def get_ray_args(parser):
    """ add ray arguments here """
    group = parser.add_argument_group('Ray configuration')
    group.add_argument('--address', type=str, default=None)
    group.add_argument('--redis_address', type=str, default=None)
    group.add_argument('--num_cpus', type=int, default=os.cpu_count())
    group.add_argument('--num_gpus', type=int, default=None)
