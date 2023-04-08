
# -- helpers --
import copy
dcopy = copy.deepcopy
import numpy as np
import torch as th
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- io --
from dev_basics import arch_io

# -- network --
from .net import get_model

# -- configs --
from dev_basics.configs import ExtractConfig,dcat
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction

# -- load the model --
@econfig.set_init
def load_model(cfg):

    # -- config --
    econfig.init(cfg)
    device = econfig.optional(cfg,"device","cuda:0")

    # -- unpack local vars --
    local_pairs = {"io":io_pairs(),
                   "arch":arch_pairs()}
    cfgs = econfig.extract_dict_of_pairs(cfg,local_pairs,restrict=True)
    cfg = dcat(cfg,econfig.flatten(cfgs)) # update cfg

    # -- end init --
    if econfig.is_init: return

    # -- load model --
    model = get_model(cfg.arch_subname)
    model = model.to(device)
    model.eval()

    # -- load model --
    load_pretrained(model,cfgs.io)

    # -- test --
    # rand_vid = th.randn((1,5,3,64,64))
    # flows = {"fflow":th.randn((1,5,2,64,64)),"bflow":th.randn((1,5,2,64,64))}
    # out = model(rand_vid,flows=flows)

    return model

def load_pretrained(model,cfg):
    if cfg.pretrained_load:
        print("Loading model: ",cfg.pretrained_path)
        cfg.pretrained_root = cfg.pretrained_root
        arch_io.load_checkpoint(model,cfg.pretrained_path,
                                cfg.pretrained_root,cfg.pretrained_type)

# -- default pairs --
def arch_pairs():
    pairs = {"arch_subname":"vit","vit_mode":"default"}
    return pairs

def io_pairs():
    base = Path("weights/checkpoints/")
    pretrained_path = base / "model/model_best.pt"
    pairs = {"pretrained_load":False,
             "pretrained_path":str(pretrained_path),
             "pretrained_type":"lit",
             "pretrained_root":"."}
    return pairs

