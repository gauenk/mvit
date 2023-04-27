

# -- misc --
import os,math,tqdm,sys
import pprint,copy
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
from dev_basics import flow

# -- caching results --
import cache_io

# # -- network --
# import nlnet
# from detectron2.structures import Instances
from detectron2.utils.events import EventStorage

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__)
extract_config = econfig.extract_config

# -- misc --
from dev_basics.utils.misc import rslice,write_pickle,read_pickle
from dev_basics.utils.metrics import compute_psnrs,compute_ssims
from dev_basics.utils.timer import ExpTimer
import dev_basics.utils.gpu_mem as gpu_mem

# -- noise sims --
import importlib
# try:
#     import stardeno
# except:
#     pass

# -- generic logging --
import logging
logging.basicConfig()

# -- lightning module --
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

# import torch
# torch.autograd.set_detect_anomaly(True)

@econfig.set_init
def init_cfg(cfg):
    econfig.init(cfg)
    cfgs = econfig({"lit":lit_pairs(),
                    "sim":sim_pairs()})
    return cfgs

def lit_pairs():
    pairs = {"batch_size":1,"flow":True,"flow_method":"cv2",
             "isize":None,"bw":False,"lr_init":1e-3,
             "lr_final":1e-8,"weight_decay":0.,
             "nepochs":0,"task":"denoising","uuid":"",
             "scheduler":"default","step_lr_size":5,
             "step_lr_gamma":0.1,"flow_epoch":None,"flow_from_end":None}
    return pairs

def sim_pairs():
    pairs = {"sim_type":"g","sim_module":"stardeno",
             "sim_device":"cuda:0","load_fxn":"load_sim"}
    return pairs

def get_sim_model(self,cfg):
    if cfg.sim_type == "g":
        return None
    elif cfg.sim_type == "stardeno":
        module = importlib.load_module(cfg.sim_module)
        return module.load_noise_sim(cfg.sim_device,True).to(cfg.sim_device)
    else:
        raise ValueError(f"Unknown sim model [{sim_type}]")

class LitModel(pl.LightningModule):

    def __init__(self,lit_cfg,net,sim_model):
        super().__init__()
        lit_cfg = init_cfg(lit_cfg).lit
        for key,val in lit_cfg.items():
            if key in ["device"]: continue
            setattr(self,key,val)
        self.set_flow_epoch() # only for current exps; makes last 10 epochs with optical flow.
        self.net = net
        self.net.train(True)
        self.sim_model = sim_model
        self.gen_loger = logging.getLogger('lightning')
        self.gen_loger.setLevel("NOTSET")
        self.automatic_optimization=True

    def forward(self,vid,flows=None,gt_annos=None):
        if flows is None:
            flows = flow.orun(vid,self.flow,ftype=self.flow_method)
        if not(gt_annos is None): self.net.train(True)
        preds = self.net(vid,flows=flows,gt_annos=gt_annos)
        return preds

    def set_flow_epoch(self):
        if not(self.flow_epoch is None): return
        if self.flow_from_end is None: return
        if self.flow_from_end == 0: return
        self.flow_epoch = self.nepochs - self.flow_from_end

    def update_flow(self):
        if self.flow_epoch is None: return
        if self.flow_epoch <= 0: return
        if self.current_epoch >= self.flow_epoch:
            self.flow = True

    def configure_optimizers(self):
        optim = th.optim.Adam(self.parameters(),lr=self.lr_init,
                              weight_decay=self.weight_decay)
        sched = self.configure_scheduler(optim)
        return [optim], [sched]

    def configure_scheduler(self,optim):
        if self.scheduler in ["default","exp_decay"]:
            gamma = 1-math.exp(math.log(self.lr_final/self.lr_init)/self.nepochs)
            ExponentialLR = th.optim.lr_scheduler.ExponentialLR
            scheduler = ExponentialLR(optim,gamma=gamma) # (.995)^50 ~= .78
        elif self.scheduler in ["step","steplr"]:
            args = (self.step_lr_size,self.step_lr_gamma)
            print("[Scheduler]: StepLR(%d,%2.2f)" % args)
            StepLR = th.optim.lr_scheduler.StepLR
            scheduler = StepLR(optim,step_size=self.step_lr_size,
                               gamma=self.step_lr_gamma)
        elif self.scheduler in ["cos"]:
            CosAnnLR = th.optim.lr_scheduler.CosineAnnealingLR
            T0,Tmult = 1,1
            scheduler = CosAnnLR(optim,T0,Tmult)
        elif self.scheduler in ["none"]:
            StepLR = th.optim.lr_scheduler.StepLR
            scheduler = StepLR(optim,step_size=10**3,gamma=1.)
        else:
            raise ValueError(f"Uknown scheduler [{self.scheduler}]")
        return scheduler

    def training_step(self, batch, batch_idx):

        # -- update flow --
        self.update_flow()

        # -- each sample in batch --
        loss = 0 # init @ zero
        denos,cleans = [],[]
        ntotal = len(batch)
        nbatch = 1#ntotal
        nbatches = (ntotal-1)//nbatch+1
        for i in range(nbatches):
            start,stop = i*nbatch,min((i+1)*nbatch,ntotal)
            loss_i = self.training_step_i(batch, start, stop)
            loss += loss_i
        loss = loss / nbatches

        # -- log --
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False,batch_size=self.batch_size)

        # -- terminal log --
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False, batch_size=self.batch_size)

        return loss

    def training_step_i(self, batch, start, stop):

        # -- unpack batch
        video = batch[start]['video']/255.
        fflow = batch[start]['fflow']
        bflow = batch[start]['bflow']
        annos = batch[start]['instances']

        # -- make flow --
        if fflow.shape[-2:] == video.shape[-2:]:
            flows = edict({"fflow":fflow,"bflow":bflow})
        else:
            flows = None

        # -- foward --
        with EventStorage(0):
            loss = self.forward(video,flows,annos)

        # -- loss --
        loss_num = sum([loss[k] for k in loss])
        return loss_num

    def validation_step(self, batch, batch_idx):

        # -- denoise --
        video = batch[0]['video']/255.
        annos = batch[0]['instances']

        # -- flow --
        fflow = batch[0]['fflow']
        bflow = batch[0]['bflow']
        if fflow.shape[-2:] == video.shape[-2:]:
            flows = edict({"fflow":fflow,"bflow":bflow})
        else:
            flows = None

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with EventStorage(0):
            loss = self.forward(video,flows,annos)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- report --
        val_loss = sum([v.item() for k,v in loss.items()])
        # print("val_loss: ",val_loss)
        self.log("val_loss", val_loss, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        for field in loss:
            # print(loss[field].item())
            self.log("val_%s"%field, loss[field].item(), on_step=False,
                     on_epoch=True, batch_size=1, sync_dist=True)
        self.log("val_mem_res", mem_res, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_mem_alloc", mem_alloc, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)

        # -- terminal log --
        # val_psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        # self.gen_loger.info("val_psnr: %2.2f" % val_psnr)

    def test_step(self, batch, batch_nb):

        # -- sample noise from simulator --
        # self.sample_noisy(batch)

        # -- denoise --
        # print(len(batch))
        # print(type(batch))
        # print(len(batch))
        # print(list(batch[0].keys()))
        index = batch[0]['image_index']
        video = batch[0]['video']/255.
        annos = batch[0]['instances']
        # annos = []
        # print(type(batch['instances'][0]))
        # print(batch['instances'][0])
        # for t in range(video.shape[0]):
        #     annos += [Instances(video.shape[-2:],**batch['instances'][t])]
        # print(": ",type(batch['instances']),type(batch['instances'][0]))

        # annos = [j for i in annos for j in i]
        # print("video.shape: ",video.shape)
        # print(len(annos))
        # for anno in annos:
        #     print(type(anno),len(anno))
        #     for anno_i in anno:
        #         print(type(anno_i),len(anno_i))
        # print(type(annos[0][0]))
        # print(len(annos[0]))
        # exit(0)

        # -- flow --
        fflow = batch[0]['fflow']
        bflow = batch[0]['bflow']
        if fflow.shape[-2:] == video.shape[-2:]:
            flows = edict({"fflow":fflow,"bflow":bflow})
        else:
            flows = None

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"test",reset=True)
        with EventStorage(0):
            loss = self.forward(video,flows,annos)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"test",reset=True)

        # -- loss --
        # print(list(preds[0]['instances'].keys()))
        # print(annos_exist)
        # B,T = video.shape[:2]
        # for b in range(B):
        #     for t in range(T):
        #         if annos_exist[b][t] != True: continue
        #         loss += th.mean((preds[b][t] - annos[b][t])**2)

        # -- terminal log --
        for field in loss:
            self.log(field, loss[field].item(), on_step=True,
                     on_epoch=False, batch_size=1)
        self.log("mem_res",  mem_res, on_step=True, on_epoch=False, batch_size=1)
        self.log("mem_alloc",  mem_alloc, on_step=True, on_epoch=False, batch_size=1)

        # -- log --
        results = edict()
        for field in loss:
            results["test_%s"%field] = loss[field].item()
        results.test_mem_alloc = mem_alloc
        results.test_mem_res = mem_res
        results.test_index = index#.cpu().numpy().item()
        return results

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def _accumulate_results(self,each_me):
        for key,val in each_me.items():
            if not(key in self.metrics):
                self.metrics[key] = []
            if hasattr(val,"ndim"):
                ndim = val.ndim
                val = val.cpu().numpy().item()
            self.metrics[key].append(val)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        print("logging metrics: ",metrics,step)

    def on_train_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_train_batch_end(self, trainer, pl_module, outs,
                           batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)


    def on_validation_batch_end(self, trainer, pl_module, outs,
                                batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_batch_end(self, trainer, pl_module, outs,
                          batch, batch_idx, dl_idx):
        self._accumulate_results(outs)



def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]
