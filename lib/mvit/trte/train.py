

import logging
import copy
dcopy = copy.deepcopy

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
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

logger = logging.getLogger("detectron2")

# -- local --
import mvit
from .test import do_test

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
    print(cfg)
    pairs = {"lr_multiplier":1.,
             "chkpt_period":300,
             "chkpt_nkeep":11,
             "log_period":10,
             "eval_period":500,
             "niters":100,}
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
    train_loader = get_dataloader(cfg)

    # -- ddp --
    model = create_ddp_model(model)#, **cfg.train.ddp)

    # -- create trainer -
    trainer = (AMPTrainer if cfg.use_amp else SimpleTrainer)(model, train_loader, optim)

    # -- checkpoint --
    checkpointer = DetectionCheckpointer(
        model,cfg.chkpt_root,trainer=trainer,
    )


    # -- hooks --
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, cfg.chkpt_period,
                                       cfg.nepochs,cfg.chkpt_nkeep,
                                       cfg.subdir)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.log_root, cfg.nepochs),
                period=cfg.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    # -- resume --
    # checkpointer.resume_or_load(cfg.pretrained_path,cfg.pretrained_load)
    # checkpointer.resume_or_load(cfg.pretrained_path,cfg.pretrained_load)
    # if cfg.pretrained_load and checkpointer.has_checkpoint():
    #     # The checkpoint stores the training iteration that just finished, thus we start
    #     # at the next iteration
    #     start_iter = trainer.iter + 1
    # else:
    #     start_iter = 0
    start_iter = 0

    # -- init --
    trainer.train(start_iter, cfg.niters)

def get_dataloader(cfg):
    loader = None
    return loader

def get_optimizer(cfg,model):
    optimizer = None
    return optimizer
