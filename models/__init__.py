# #!/usr/bin/env python
#
# # -*- encoding: utf-8
#
# '''
#     _____.___._______________  __.____ __________    _________   ___ ___    _____  .___
#     \__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
#      /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
#      \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
#      / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
#      \/               \/        \/               \/          \/       \/         \/
#
#  ==========================================================================================
#
# @author: Yekun Chai
#
# @license: School of Informatics, Edinburgh
#
# @contact: chaiyekun@gmail.com
#
# @file: __init__.py.py
#
# @time: 20/05/2020 14:10
#
# @desc：
#
# '''
import os
import importlib
#
MODEL_REGISTRY = {}
#
#
def setup_model(args):
    return MODEL_REGISTRY[args.model_name]


def register_model(name):
    """
    register your custom model before class
    :param name:
    :return:
    """
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Model already registered!")
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls

# import sys
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith(".") and (file.endswith('.py') or os.path.isdir(path)):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'models.{model_name}')
