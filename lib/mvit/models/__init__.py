

import importlib

def extract_model_config(cfg):
    return extract_config(cfg)

def extract_config(cfg):
    task = optional(cfg,'task','segm')
    return importlib.import_module("from mvit.models import %s"%task).extract_config(cfg)

def load_model(cfg):
    task = optional(cfg,'task','segm')
    return importlib.import_module("from mvit.models import %s"%task).load_model(cfg)
