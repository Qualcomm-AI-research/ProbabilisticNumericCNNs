# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn.functional as F
from image_remeshing_cnn.kernel import LinearOperatorGP,HeatRBF
import image_remeshing_cnn
from image_remeshing_cnn.trainer import PNCNNTrainer
from oil.model_trainers import Trainer,Classifier
from image_remeshing_cnn.architecture import PNCNN
import os
import torch
from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, islice, Eval
from oil.datasetup.datasets import split_dataset
from functools import partial
from torch.optim import Adam, SGD
import numpy as np
import torchvision
from oil.utils.utils import Expression
from collections import defaultdict
import pandas as pd
from oil.utils.mytqdm import tqdm
import torch.nn as nn

def points2img(x):
    coords,vals = x
    bs,n,c = vals.shape
    sqrtn = int(np.sqrt(n))
    return vals.permute(0,2,1).reshape(bs,c,sqrtn,sqrtn)

class interpolatedMNIST(torchvision.datasets.MNIST):
    class_weights=None
    balanced=True
    stratify=True
    ignored_index=-100
    num_targets = 10
    def __init__(self,*args,**kwargs):
        super().__init__(*args,download=True,**kwargs)
        self.coords = torch.from_numpy((np.mgrid[:28,:28]/28).reshape(2,-1).T).float()
    def __getitem__(self,idx):
        N = self.coords.shape[0]
        img = (self.data[idx].float()/255.-.1307)/0.3081
        return img[None], int(self.targets[idx])

def ConvBNrelu(in_channels,out_channels,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        k=128
        self.network = nn.Sequential(
            ConvBNrelu(1,k),
            ConvBNrelu(k, k),
            nn.MaxPool2d(2),
            ConvBNrelu(k, k),
            ConvBNrelu(k, k),
            nn.MaxPool2d(2),
            ConvBNrelu(k, k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(k,10),
        )
    def forward(self,x):
        return self.network(x)


def makeTrainer(*, train_res=9,dataset=interpolatedMNIST, network=CNN, num_epochs=20,
                bs=50, lr=3e-3, optim=Adam, device='cuda',trainer_config={},
                split={'train': 60000}):

    res = train_res
    # Prep the datasets splits, model, and dataloaders
    datasets = split_dataset(dataset('./datasets/interpolatedMNIST'),splits=split)
    datasets['test'] = dataset('./datasets/interpolatedMNIST', train=False)
    device = torch.device(device)
    model = network().to(device)
    model = torch.nn.Sequential(Expression(lambda x: F.interpolate(x,(res,res),mode="bilinear")),model)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    opt_constr = partial(optim, lr=lr)
    lr_sched = cosLr(num_epochs)
    return Classifier(model,dataloaders,opt_constr=opt_constr,lr_sched=lr_sched,**trainer_config)

resolutions = [4,6,8,9,10,12,14,16,18]
results_df = pd.DataFrame({'test_res':[],'train_res':[],'test_acc':[]})
for train_res in tqdm(resolutions,desc="train_res"):
    trainer = makeTrainer(train_res=train_res,trainer_config=
    {'log_dir':os.path.expanduser('~/ws/tensorboard/'),'log_suffix':f'res_expt_cnn{train_res}'})
    trainer.train(20)
    torch.cuda.empty_cache()
    results = defaultdict(list)
    for test_res in tqdm(resolutions,desc="test_res"):
        trainer.model[0] = Expression(lambda x: F.interpolate(x,(test_res,test_res),mode="bilinear"))
        results['test_res'].append(test_res)
        results['train_res'].append(train_res)
        results['test_acc'].append(trainer.metrics(trainer.dataloaders['test'])['Acc'])
    results_df = results_df.append(pd.DataFrame(results),ignore_index=True)
    results_df.to_pickle("./res_test_cnn_df.pkl")
