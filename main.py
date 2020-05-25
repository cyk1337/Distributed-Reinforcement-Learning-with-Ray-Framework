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

@file: main.py

@time: 20/05/2020 14:14 

@desc：       
               
'''
import os
# import ray
from rl.utils import *
from rl import options, learners
# To be customized
from rl.rollout_worker import DistRolloutWorkerPool

if __name__ == '__main__':
    parser = options.get_training_parser(default_env='gym_env',
                                         default_model='ppo',
                                         default_player='default_player',
                                         default_learner='default_learner',
                                         )
    args = options.parse_custom_args(parser)
    print(args)

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not getattr(args, 'num_gpus'):
        args.num_gpus = 2

    os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

    ray.init(address=args.address, redis_address=args.redis_address, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
             include_webui=False, ignore_reinit_error=True)
    assert ray.is_initialized()

    learner = learners.setup_learner(args)

    rollout_worker = DistRolloutWorkerPool(learner, args)
    print(f'Start running [{rollout_worker}] ...')
    states = rollout_worker.start()
    player = rollout_worker.get_player()
    while True:
        actions = player.step(states)
        states = rollout_worker.run(actions)
        learner.run()
