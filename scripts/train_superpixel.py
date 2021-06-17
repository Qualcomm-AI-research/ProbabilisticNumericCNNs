# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
import os
import torch
from image_remeshing_cnn.datasets import MNISTSuperpixels
from image_remeshing_cnn.architecture import PNCNN
from image_remeshing_cnn.trainer import PNCNNTrainer
from oil.model_trainers import Classifier
import oil.model_trainers as trainers
from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, islice
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from functools import partial
from torch.optim import Adam, SGD
from oil.tuning.args import argupdated_config
import image_remeshing_cnn.kernel as kernel

def makeTrainer(*, dataset=MNISTSuperpixels, network=PNCNN, num_epochs=20,data_dir=None,
                bs=50, lr=3e-3, optim=Adam, device='cuda', trainer=PNCNNTrainer,split={'train':60000},
                net_config={'k':128,'num_layers':4,'kernel':kernel.HeatRBF,'num_basis':9,'nounc':False},
                opt_config={},trainer_config={'gp_weight':1/200.,'log_dir':os.path.expanduser('~/ws/tensorboard/'),
                                'log_suffix':'pncnn_standard_new_runs_nounc'}):
    if net_config['nounc']:
        trainer_config.pop('gp_weight')
        trainer = Classifier
    # Prep the datasets splits, model, and dataloaders
    datasets = split_dataset(dataset(data_dir),splits=split)
    datasets['test'] = dataset(data_dir, train=False)
    device = torch.device(device)
    model = network(num_targets=datasets['train'].num_targets,**net_config).to(device)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10) # subsampled for logging
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs) # Learning rate schedule with cosine decay
    return trainer(model,dataloaders,opt_constr=opt_constr,lr_sched=lr_sched,**trainer_config)

simpleTrial = train_trial(makeTrainer)
if __name__=='__main__':
    simpleTrial(argupdated_config(makeTrainer.__kwdefaults__,namespace=(kernel,trainers)))
    # kwargs = makeTrainer.__kwdefaults__
    # for i in range(3):
    #     kwargs['trainer_config']['log_suffix'] = kwargs['trainer_config']['log_suffix'][:-1] + str(i)
    #     simpleTrial(argupdated_config(kwargs, namespace=(kernel, trainers)))
