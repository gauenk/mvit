

import importlib
from dev_basics.utils.misc import optional

def extract_model_config(cfg):
    return extract_config(cfg)

def extract_config(cfg):
    task = optional(cfg,'task','segm')
    return importlib.import_module("mvit.models.%s"%task).extract_config(cfg)

def load_model(cfg):
    task = optional(cfg,'task','segm')
    return importlib.import_module("mvit.models.%s"%task).load_model(cfg)
