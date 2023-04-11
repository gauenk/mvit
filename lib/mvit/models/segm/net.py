
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

def get_model(arch_subname):

    # -- init arch subname --
    if arch_subname.startswith("vit"):
        model_cfg = get_vit_model(arch_subname)
    elif arch_subname.startswith("mvit"):
        model_cfg = get_mvit_model(arch_subname)
    else:
        raise ValueError(f"Uknown arch_subname [{arch_subname}]")

    # -- create model --
    # print(arch_subname)
    # print(model_cfg)
    model = instantiate(model_cfg) # config -> model
    # print(model)
    # exit(0)

    # -- get attn layers --
    attn_layers = []
    blocks = model.backbone.net.blocks
    for name,child in model.backbone.net.blocks.named_children():
        attn_layers.append(getattr(blocks,name).attn)

    # -- wrap forward model --
    model.flows = None
    _forward = model.forward
    def forward(self,in_dict,flows=None):

        # -- set flows --
        for attn in attn_layers:
            attn.flows = flows

        # keys: file_name, height, width, image_id, image
        # # -- format input --
        # B = vid.shape[0]
        # vid = rearrange(vid,'b t c h w -> (b t) c h w')*255.
        # # print("vid.shape: ",vid.shape)
        # in_dict = [{"image":vid[b]} for b in range(B)]

        # -- get preds --
        preds_raw = _forward(in_dict)
        # print(preds_raw)
        # import pickle
        # with open("tmp.pkl","wb") as f:
        #     pickle.dump(preds_raw,f)

        # # -- transpose preds --
        # # keys = ["instances","pred_classes","pred_masks","pred_boxes","scores"]
        # keys1 = list(preds_raw[0]['instances'].keys())
        # # keys1 = ["instances","pred_classes","pred_masks","pred_boxes","scores"]
        # preds = {}
        # for key in keys:
        #     preds[key] = [preds_raw[b]['instances'][key] for b in range(B)]
        #     if hasattr(preds[key],"shape"):
        #         print(key,preds[key].shape)
        #     else:
        #         print(key,len(preds[key]))
        return preds_raw

    # model.forward = partial(forward,model)
    model = model.train()

    return model

def get_vit_model(arch_subname):

    # -- load base --
    # arch_subname
    # model = model_zoo.get_config("common/models/mask_rcnn_vitstnls.py").model
    model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

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
        # 3,7 for global attention
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

def get_mvit_model(arch_subname):
    model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
    constants = model_zoo.get_config("common/data/constants.py").constants
    model.pixel_mean = constants.imagenet_rgb256_mean
    model.pixel_std = constants.imagenet_rgb256_std
    model.input_format = "RGB"
    model.backbone.bottom_up = L(MViT)(
        embed_dim=96,
        depth=10,
        num_heads=1,
        last_block_indexes=(0, 2, 7, 9),
        residual_pooling=True,
        drop_path_rate=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        out_features=("scale2", "scale3", "scale4", "scale5"),
    )
    model.backbone.in_features = "${.bottom_up.out_features}"
    return model

