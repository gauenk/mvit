"""

Unsupervised Training with Frame2Frame

Compare the impact of train/test using exact/refineimate methods

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from dev_basics.trte import train

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
    results = cache_io.run_exps(exps,train.run,uuids=uuids,
                                name=".cache_io/baseline/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/baseline/train.pkl",
                                records_reload=True)

if __name__ == "__main__":
    main()
