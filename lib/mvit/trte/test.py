
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- dev basics --
# from dev_basics.report import deno_report
from functools import partial
from dev_basics.aug_test import test_x8
from dev_basics import flow
from dev_basics import net_chunks
from dev_basics.utils.misc import get_region_gt
from dev_basics.utils.misc import optional,slice_flows,set_seed
from dev_basics.utils.misc import rslice,write_pickle,read_pickle
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.metrics import compute_psnrs,compute_ssims,compute_strred
from dev_basics.utils import vid_io

# -- segmentation eval --
from detectron2.evaluation import SemSegEvaluatorV2

# -- config --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__)
extract_config = econfig.extract_config

def test_pairs():
    pairs = {"device":"cuda:0","seed":123,
             "frame_start":0,"frame_end":-1,
             "aug_test":False,"longest_space_chunk":False,
             "flow":False,"burn_in":False,"arch_name":None,
             "saved_dir":"./output/saved_examples/","uuid":"uuid_def",
             "flow_sigma":-1,"internal_adapt_nsteps":0,
             "internal_adapt_nepochs":0,"nframes":0,"read_flows":False,
             "save_deno":True,"python_module":"dev_basics.trte.id_model",
             "bench_bwd":False,"append_noise_map":False,"arch_name":"default"}
    return pairs

@econfig.set_init
def run(cfg):

    # -- config --
    econfig.init(cfg)
    epairs = econfig.extract_pairs
    tcfg = epairs(cfg,test_pairs())
    module = econfig.required_module(tcfg,'python_module')
    model_cfg = epairs(module.extract_model_config(tcfg),cfg)
    if econfig.is_init: return
    if tcfg.frame_end == -1: tcfg.frame_end = tcfg.frame_start + cfg.nframes - 1

    # -- clear --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    # -- set seed --
    set_seed(tcfg.seed)

    # -- set device --
    th.cuda.set_device(int(tcfg.device.split(":")[1]))

    # # -- init results --
    # results = edict()
    # results.examples = []

    # # -- init keyword fields --
    # time_fields = ["flow","deno","attn","extract","search",
    #                "agg","fold","fwd_grad","bwd"]
    # for field in time_fields:
    #     results["timer_%s"%field] = []
    # mem_fields = ["deno","adapt","fwd_grad","bwd"]
    # for field in mem_fields:
    #     results["%s_mem_res"%field] = []
    #     results["%s_mem_alloc"%field] = []

    # -- burn_in once --
    burn_in = tcfg.burn_in

    # -- load model --
    model = module.load_model(model_cfg)

    # -- augmented testing --
    if tcfg.aug_test:
        aug_fxn = partial(test_x8,model)#,use_refine=cfg.aug_refine_inds)
    else:
        aug_fxn = model.forward

    # -- chunked processing --
    chunk_cfg = net_chunks.extract_chunks_config(cfg)
    if tcfg.longest_space_chunk:
        set_longest_spatial_chunk(chunk_cfg,noisy.shape)
    fwd_fxn = net_chunks.chunk(chunk_cfg,aug_fxn)
    chunk_fwd = fwd_fxn

    # -- eval --
    output_dir = Path(tcfg.saved_dir) / tcfg.arch_name / str(tcfg.uuid)
    seg_eval = SemSegEvaluatorV2(tcfg.dname,output_dir=output_dir)

    # -- iterate over data --
    for sample in loaders[tcfg.dset]:

        # -- unpack --
        vid = sample['vid'].to(device)
        anno = sample['anno']

        # -- forward --
        with th.no_grad():
            preds = model(sample['vid'])

        # -- eval --
        seg_eval.process(anno,preds)

    # -- finalize --
    res = seg_eval.evaluate()['sem_seg']
    return edict(res)


