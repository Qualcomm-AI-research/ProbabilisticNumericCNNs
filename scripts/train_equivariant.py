# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
import os
import torch
from image_remeshing_cnn.datasets import MNISTSuperpixels
from image_remeshing_cnn.equivariant.equivariant import GPNCNN
from image_remeshing_cnn.trainer import PNCNNTrainer
import oil.model_trainers as trainers
from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, islice
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from functools import partial
from torch.optim import Adam, SGD
from oil.tuning.args import argupdated_config
import image_remeshing_cnn.kernel as kernel
#import RBF,RBFwBlur,HeatRBF

def makeTrainer(*, dataset=MNISTSuperpixels, network=GPNCNN, num_epochs=20,
                bs=50, lr=3e-3, aug=True, optim=Adam, device='cuda', trainer=PNCNNTrainer,
                split={'train':60000},data_dir=None,
                net_config={},
                #net_config={'nounc':True},
                opt_config={},trainer_config={'gp_weight':1/200.,'log_dir':os.path.expanduser('~/ws/tensorboard/'),
                                'log_suffix':'gpncnn_MLL200_large'}):

    # Prep the datasets splits, model, and dataloaders
    datasets = split_dataset(dataset(data_dir),splits=split)
    datasets['test'] = dataset(data_dir, train=False)
    device = torch.device(device)
    model = network(num_targets=datasets['train'].num_targets,**net_config).to(device)
    if aug: model = torch.nn.Sequential(datasets['train'].default_aug_layers(),model)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10)
    # Add some extra defaults if SGD is chosen
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)
    return trainer(model,dataloaders,opt_constr=opt_constr,lr_sched=lr_sched,**trainer_config)


simpleTrial = train_trial(makeTrainer)
if __name__=='__main__':
    simpleTrial(argupdated_config(makeTrainer.__kwdefaults__,namespace=(kernel,trainers)))
