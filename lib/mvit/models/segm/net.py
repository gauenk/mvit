
# -- import detectron --
from functools import partial
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler
from einops import rearrange

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import MViT
from detectron2.config import LazyConfig, instantiate
from pathlib import Path

def get_model(cfgs):

    # -- load base --
    attn_type = cfgs.arch.attn_type
    if cfgs.arch.attn_type == "default":
        model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
    else:
        model = model_zoo.get_config("common/models/mask_rcnn_vitstnls.py").model

    # -- select network --
    keys = ["embed_dim","depth","num_heads","drop_path_rate"]
    for key in keys:
        setattr(model.backbone.net,key,cfgs.arch[key])

    # -- set non-local search info --
    if attn_type != "default":
        model.backbone.net.in_feature = cfgs.arch['in_feature']
        setattr(model.backbone.net,"arch_cfg",cfgs.arch)
        setattr(model.backbone.net,"attn_cfg",cfgs.attn)
        setattr(model.backbone.net,"search_cfg",cfgs.search)
        setattr(model.backbone.net,"normz_cfg",cfgs.normz)
        setattr(model.backbone.net,"agg_cfg",cfgs.agg)

    # -- allow for global if default search --
    if attn_type == "default":
        model.backbone.net.window_block_indexes = (
            list(range(0, 2))
            + list(range(4, 6))
            + list(range(8, 9))
        )

    # -- load model --
    model = instantiate(model)
    # if attn_type == "default":
    #     _forward = model.forward
    #     def forward(vid,flows=None):
    #         vid = rearrange(vid,'b t c h w -> (b t) c h w')
    #         out = _forward(vid)
    #     model.forward = forward

    return model

