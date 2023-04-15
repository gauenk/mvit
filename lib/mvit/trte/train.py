

import logging
import copy
dcopy = copy.deepcopy
from pathlib import Path

import numpy as np
import torch as th

import data_hub
from .coco import dataloader

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
# from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2 import model_zoo
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

logger = logging.getLogger("detectron2")

# -- local --
import mvit
# from .test import do_test

def run(cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """

    # -- view --
    pairs = {"lr_multiplier":1.,
             "chkpt_period":300,
             "chkpt_nkeep":11,
             "log_period":10,
             "eval_period":500,
             "niters":10000,
             "use_amp":False}
    cfg = dcopy(cfg)
    for key,val in pairs.items():
        cfg[key] = val
    cfg.max_iter_period = cfg.nepochs

    # -- init logger --
    logger = logging.getLogger("detectron2")

    # -- load model --
    model = mvit.load_model(cfg)
    logger.info("Model:\n{}".format(model))
    model.to(cfg.device)

    # -- init optimizer --
    # cfg.optimizer.params.model = model
    optim = get_optimizer(cfg,model)
    # optim = instantiate(cfg.optimizer)

    # -- init data loader --
    # train_loader = get_dataloader(cfg)
    # print(train_loader)
    # print(list(dataloader.keys()))
    # train_loader = dataloader['train'].dataset.names
    dataloader.train.total_batch_size = 1
    train_loader = instantiate(dataloader.train)
    # print(dir(dataloader['train']))
    # print(dir(dataloader['train'].dataset))
    # print(type(train_loader))

    # -- train info --
    train_info = get_train_info()

    # -- lr-mult --
    lr_mult = get_lr_mult(cfg.niters)

    # -- ddp --
    model = create_ddp_model(model, **train_info.ddp)

    # -- create trainer -
    trainer = (AMPTrainer if cfg.use_amp else SimpleTrainer)(model, train_loader, optim)

    # -- checkpoint --
    checkpointer = DetectionCheckpointer(
        model,cfg.chkpt_root,trainer=trainer,
    )
    if not Path(cfg.chkpt_root).exists():
        Path(cfg.chkpt_root).mkdir(parents=True)

    # -- hooks --
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(lr_mult)),
            hooks.PeriodicCheckpointer(checkpointer, **train_info.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(train_info.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(train_info.output_dir, train_info.max_iter),
                period=train_info.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )
    # print(trainer)
    # trainer.register_hooks(
    #     [
    #         hooks.IterationTimer(),
    #         hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
    #         hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
    #         if comm.is_main_process()
    #         else None,
    #         hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
    #         hooks.PeriodicWriter(
    #             default_writers(cfg.train.output_dir, cfg.train.max_iter),
    #             period=cfg.train.log_period,
    #         )
    #         if comm.is_main_process()
    #         else None,
    #     ]
    # )

    # trainer.register_hooks(
    #     [
    #         hooks.IterationTimer(),
    #         hooks.LRScheduler(scheduler=instantiate(lr_mult)),
    #         hooks.PeriodicCheckpointer(checkpointer, cfg.chkpt_period,
    #                                    cfg.niters,cfg.chkpt_nkeep,
    #                                    cfg.subdir)
    #         if comm.is_main_process()
    #         else None,
    #         hooks.EvalHook(cfg.eval_period, lambda: do_test(cfg, model)),
    #         hooks.PeriodicWriter(
    #             default_writers(cfg.log_root, cfg.niters),
    #             period=cfg.log_period,
    #         )
    #         if comm.is_main_process()
    #         else None,
    #     ]
    # )

    # -- resume --
    params = list(model.parameters())
    L = len(params)
    # params_og = [params[i].clone() for i in range(len(params))]
    # checkpointer.resume_or_load(train_info.init_checkpoint, resume=False)
    # checkpointer.resume_or_load(cfg.pretrained_path,cfg.pretrained_load)
    # checkpointer.resume_or_load(cfg.pretrained_path,cfg.pretrained_load)
    # if cfg.pretrained_load and checkpointer.has_checkpoint():
    #     # The checkpoint stores the training iteration that just finished, thus we start
    #     # at the next iteration
    #     start_iter = trainer.iter + 1
    # else:
    #     start_iter = 0
    start_iter = 0
    # print(np.mean([th.mean((params_og[i] - params[i])**2).item() for i in range(L)]))

    # exit(0)

    # -- init --
    trainer.train(start_iter, cfg.niters)

def get_dataloader(cfg):
    data,loaders = data_hub.sets.load(cfg)
    return loaders.tr

def get_lr_mult(max_iter):
    lr_multiplier = L(WarmupParamScheduler)(
        scheduler=L(MultiStepParamScheduler)(
            # values=[1.0, 0.1, 0.01],
            values=[1.0, 0.1, 0.01],
            # milestones=[163889, 177546],
            milestones=[50,100],
            num_updates=max_iter,
        ),
        warmup_length=250 / max_iter,
        warmup_factor=0.001,
    )
    return lr_multiplier

def get_optimizer(cfg,model):
    optim = th.optim.Adam(model.parameters(),lr=cfg.lr_init,
                          weight_decay=cfg.weight_decay)

    return optim

def get_train_info():
    train = model_zoo.get_config("common/train.py").train
    train.amp.enabled = True
    train.ddp.fp16_compression = True
    train.init_checkpoint = (
        "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_large.pth?matching_heuristics=True"
    )
    # train.init_checkpoint = (
    #     "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
    # )

    # Schedule
    # 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
    train.max_iter = 184375
    return train

# from detectron2.config import LazyCall as L
# from detectron2.solver import WarmupParamScheduler
# from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

# from ..common.coco_loader_lsj import dataloader


# model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

# # Initialization and trainer settings
# train = model_zoo.get_config("common/train.py").train
# train.amp.enabled = True
# train.ddp.fp16_compression = True
# train.init_checkpoint = (
#     "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
# )


# # Schedule
# # 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
# train.max_iter = 184375

# lr_multiplier = L(WarmupParamScheduler)(
#     scheduler=L(MultiStepParamScheduler)(
#         values=[1.0, 0.1, 0.01],
#         milestones=[163889, 177546],
#         num_updates=train.max_iter,
#     ),
#     warmup_length=250 / train.max_iter,
#     warmup_factor=0.001,
# )

