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
import ray

import options
from agents import setup_agent
from models import setup_model
from worker import DefaultWorker


def cli_train():
    # Create default parser from CLI
    parser = options.get_training_parser(default_env='gym_env',
                                         default_model='ppo',
                                         default_agent='default_agent',
                                         )
    # Modify this function when customizing new arguments
    args = options.parse_custom_args(parser)
    print(args)

    ############# test #############
    # Comment out this block when running.
    # Make sure that the assigned GPU numbers is no less than the worker numbers
    args.gpu = "0,2,3"
    args.num_workers = 3
    ################################

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not args.num_gpus:
        args.num_gpus = len(args.gpu.split(','))
    assert args.num_gpus
    assert args.num_workers <= args.num_gpus

    # Init ray
    ray.init(address=args.address, redis_address=args.redis_address, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
             include_webui=False, ignore_reinit_error=True)
    assert ray.is_initialized()

    # Define agents / models
    agent_cls = ray.remote(setup_agent(args))
    model_cls = setup_model(args)
    local_model = model_cls(args, )
    remote_model_cls = ray.remote(num_gpus=1)(model_cls)

    # Create models / agents / workers
    all_models = [remote_model_cls.remote(args, ) for _ in range(args.num_workers)]
    all_agents = [agent_cls.remote(args, model) for model in all_models]
    all_workers = [DefaultWorker.remote(args, all_agents[i], all_models[i]) for i in range(args.num_workers)]

    # Get initial weights of local networks
    weights = local_model.get_weights()
    # put the weights in the object store
    weights = [ray.put(w) for w in weights]

    print(f'Start running [workers] ...')
    # Get initial observations
    all_states = [worker.start.remote() for worker in all_workers]
    while True:
        # Run agent.step
        all_actions = [agent.step.remote(states) for agent, states in zip(all_agents, all_states)]
        # Run worker.step
        all_states = [worker.step.remote(actions) for worker, actions in zip(all_workers, all_actions)]
        # Compute gradients from agents given unified weights
        rets = [agent.fetch_grads.remote(weights) for agent in all_agents]
        rets = [ret for ret in ray.get(rets) if ret is not None]
        # Check if it is trained (trajectories satisfy training batch size)
        # If so, grads of actors and critics will be returned
        # otherwise return None.
        if len(rets) > 0:
            # Collect actor / critic gradients
            a_grads, c_grads = list(zip(*rets))
            a_grads = ray.get(list(a_grads))
            c_grads = ray.get(list(c_grads))
            # Take the mean of all gradients
            avg_a_grads = [sum(g[i] for g in a_grads) / len(a_grads) for i in range(len(a_grads[0]))]
            avg_c_grads = [sum(g[i] for g in a_grads) / len(c_grads) for i in range(len(c_grads[0]))]
            # Update local networks
            a_feed_dict = {grad[0]: m_grad for (grad, m_grad) in zip(local_model.a_grads, avg_a_grads)}
            c_feed_dict = {grad[0]: m_grad for (grad, m_grad) in zip(local_model.c_grads, avg_c_grads)}
            local_model.sess.run(local_model.atrain_op, a_feed_dict)
            local_model.sess.run(local_model.ctrain_op, c_feed_dict)
            # take the updated weights
            weights = local_model.get_weights()
            weights = [ray.put(w) for w in weights]


if __name__ == '__main__':
    cli_train()
