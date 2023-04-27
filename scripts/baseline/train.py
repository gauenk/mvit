"""

Basic train

"""


# -- sys --
import os
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

# -- testing --
# from mvit.trte import train
from dev_basics.trte import train

# -- caching results --
import cache_io

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    def clear_fxn(num,cfg): return False
    exps,uuids = cache_io.train_stages.run("exps/baseline/train.cfg",
                                           ".cache_io_exps/baseline/train/")
    # 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
    # train.max_iter = 184375
    cfg = edict({
        "use_amp":False,
        "log_root":"./output/train/baseline/logs",
        "chkpt_root":"./output/train/baseline/checkpoints",
        "subdir":"baseline",
        "python_module":"mvit",
        "arch_subname":"vit_l",
        "attn_type":"stnls",
        "device":"cuda:0",

        "lr_init":0.0001,
        "weight_decay":0.,
        "isize":"128_128",
        "nsamples_te":3,
        "nsamples_val":3,
        "batch_size":1,
        "batch_size_tr":2,
        "nframes":5,
        "dname":"youtube",
        "ndevices":1,
        "num_workers":4,
        "limit_train_batches":0.20,
        "nepochs":10,
        "sigma":30,
        "ws":15,
        "wt":3,
    })
    results = cache_io.run_exps([cfg],train.run,#uuids=uuids,
                                name=".cache_io/baseline/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/baseline/train.pkl",
                                records_reload=True,use_wandb=False)

if __name__ == "__main__":
    main()

