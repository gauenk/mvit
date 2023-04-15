
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
import stnls
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
                   "arch":arch_pairs(),
                   "attn":attn_pairs()}
    cfgs = econfig.extract_dict_of_pairs(cfg,local_pairs,restrict=True)
    cfg = dcat(cfg,econfig.flatten(cfgs)) # update cfg
    cfg.nheads = cfg.num_heads
    dep_pairs = {"normz":stnls.normz.econfig,
                 "agg":stnls.agg.econfig}
    cfgs = dcat(cfgs,econfig.extract_dict_of_econfigs(cfg,dep_pairs))
    cfg = dcat(cfg,econfig.flatten(cfgs))
    cfgs.search = stnls.search.extract_config(cfg)
    cfg = dcat(cfg,econfig.flatten(cfgs))

    # -- end init --
    if econfig.is_init: return

    # -- load model --
    model = get_model(cfgs)#.arch)
    model = model.to(device)
    model.eval()

    # -- load model --
    # cfgs.io.pretrained_load = True
    # cfgs.io.pretrained_root = "../detectron2/"
    # cfgs.io.pretrained_path = "model_final_1a1c30.pkl"
    # print(cfgs.io)
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
    pairs = {"embed_dim":9,"depth":3,"num_heads":2,"drop_path_rate":0.1,
             "attn_type":"stnls","in_feature":3,
             "input_proj_depth":3,"output_proj_depth":3,
    }
    return pairs

def io_pairs():
    base = Path(".")
    pretrained_path = base / "weights/model_final_1a1c30.pkl"
    pairs = {"pretrained_load":False,
             "pretrained_path":str(pretrained_path),
             "pretrained_type":"npy",
             "pretrained_root":"."}
    return pairs

def attn_pairs():
    pairs = {"qk_frac":1.,"qkv_bias":True,
             "token_mlp":'leff',"attn_mode":"default",
             "token_projection":'linear',"embed_dim":9,"nheads":2,
             "use_state_update":False,"use_flow":True,
             "drop_rate_proj":0.,"attn_timer":False,"nres":3,"res_ksize":5}
    return pairs
