"""

Unsupervised Training with Frame2Frame

Compare the impact of train/test using exact/refineimate methods

"""


# -- sys --
import os
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

# -- testing --
from mvit.trte import train

# -- caching results --
import cache_io

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    def clear_fxn(num,cfg):
        return False
    exps,uuids = cache_io.train_stages.run("exps/baseline/train.cfg",
                                           ".cache_io_exps/baseline/train/")
    cfg = edict({
        "dname":"coco",
        "device":"cuda:0",
        "ndevices":1,
        "use_amp":False,
        "log_root":"./output/train/baseline/logs",
        "chkpt_root":"./output/train/baseline/checkpoints",
        "nepochs":10,
        "subdir":"baseline",
    })
    results = cache_io.run_exps([cfg],train.run,#uuids=uuids,
                                name=".cache_io/baseline/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/baseline/train.pkl",
                                records_reload=True)

if __name__ == "__main__":
    main()

