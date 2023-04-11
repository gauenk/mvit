

# -- import detectron --
from pathlib import Path
from functools import partial
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler
from einops import rearrange

# -- detectron2 --
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import MViT
from detectron2.config import LazyConfig, instantiate


def get_model(cfg):
    model = get_detectron2(cfg.arch_subname)
    return model

def get_detectron2(arch_subname):

    # -- load base --
    model = model_zoo.get_config("common/models/mask_rcnn_vitstnls.py").model
    # model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

    # print("[get_vid_model] arch_subname: ",arch_subname)
    if arch_subname.endswith("_l"):
        model.backbone.net.embed_dim = 1024
        model.backbone.net.depth = 24
        model.backbone.net.num_heads = 16
        model.backbone.net.drop_path_rate = 0.4
        # 5, 11, 17, 23 for global attention
        model.backbone.net.window_block_indexes = (
            list(range(0, 5)) + list(range(6, 11)) + \
            list(range(12, 17)) + list(range(18, 23))
        )
    elif arch_subname.endswith("_t"):
        model.backbone.net.embed_dim = 9
        model.backbone.net.depth = 10
        model.backbone.net.num_heads = 2
        model.backbone.net.drop_path_rate = 0.1
        # 5, 11, 17, 23 for global attention
        model.backbone.net.window_block_indexes = (
            list(range(0, 2))
            + list(range(4, 6))
            + list(range(8, 9))
        )
        model.backbone.net.attn_type = "default"
    elif arch_subname.endswith("_s"):
        model.backbone.net.embed_dim = 9
        model.backbone.net.depth = 10
        model.backbone.net.num_heads = 2
        model.backbone.net.drop_path_rate = 0.1
        model.backbone.net.attn_type = "space"
        # 5, 11, 17, 23 for global attention
    elif arch_subname.endswith("_n"):
        model.backbone.net.embed_dim = 9
        model.backbone.net.depth = 10
        model.backbone.net.num_heads = 2
        model.backbone.net.drop_path_rate = 0.1
        model.backbone.net.attn_type = "space-time"
    elif arch_subname.endswith("_h"):
        model.backbone.net.embed_dim = 1280
        model.backbone.net.depth = 32
        model.backbone.net.num_heads = 16
        model.backbone.net.drop_path_rate = 0.5
        # 7, 15, 23, 31 for global attention
        model.backbone.net.window_block_indexes = (
            list(range(0, 7)) + list(range(8, 15)) + \
            list(range(16, 23)) + list(range(24, 31))
        )

    return model
