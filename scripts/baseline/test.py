"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from dev_basics.trte import test

# -- plotting --
# import apsearch
# from apsearch import plots

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
    read_test = cache_io.read_test_config.run
    exps = read_test("exps/baseline/test.cfg",
                     cache_name=".cache_io_exps/baseline/test",
                     cache_reset=False)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/baseline/test",
                                    no_config_check=False)

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,
                                name=".cache_io/baseline/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/baseline/test.pkl",
                                records_reload=False,to_records_fast=False)

    # -- view --
    # print(len(results))
    if len(results) == 0: return
    afields = ['psnrs','ssims','strred']
    for f in afields: results[f] = results[f].apply(np.mean)
    for sigma,sdf in results[afields + ["vid_name","sigma"]].groupby("sigma"):
        print(sigma)
        print(sdf)
    # print(results[afields + ["vid_name","sigma"]])

if __name__ == "__main__":
    main()
