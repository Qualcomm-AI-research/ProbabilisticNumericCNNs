# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn as nn
from torch import optim
from oil.model_trainers import Trainer,Classifier
from oil.utils.utils import export
import pandas as pd
import numpy as np
import sklearn as sk

#for image data
@export
class PNCNNTrainer(Classifier):
    def __init__(self, *args, gp_weight=1 / 200., **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers['gp_weight'] = gp_weight


    def loss(self,minibatch):
        crossent = super().loss(minibatch)
        return crossent+self.gp_loss()*self.hypers['gp_weight']
    
    def gp_loss(self):
        """ should be called after a standard forward pass of the model"""
        return sum(m.nll() for m in self.model.modules() if hasattr(m,'nll'))

    def get_acc_mll(self,minibatch):
        x,y = minibatch
        acc = self.model(x).max(1)[1].type_as(y).eq(y).float().mean().cpu().item()
        average_nll = self.gp_loss().cpu().item()
        return pd.Series({'Acc':acc,'nll':average_nll})

    def metrics(self, loader):
        return self.evalAverageMetrics(loader,self.get_acc_mll)
    
    def logStuff(self,step,minibatch=None):
        super().logStuff(step,minibatch)

# For time series
@export
class PNCNNTrainer2(Classifier):
    def __init__(self, *args,gp_weight=1/20000., **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers['gp_weight'] = gp_weight


    def loss(self,minibatch):
        x,y = minibatch
        logit_pred = self.model(x).squeeze(-1)
        pos_weight=torch.tensor([self.dataloaders['train'].dataset.class_weights]).float().to(logit_pred.device)
        BCE_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logit_pred,y.float())
        return BCE_loss+self.gp_loss()*self.hypers['gp_weight']

    def gp_loss(self):
        """ should be called after a standard forward pass of the model"""
        return sum(m.nll() for m in self.model.modules() if hasattr(m,'nll'))
    def fresh_mll(self,mb):
        self.model(mb[0])
        return self.gp_loss().cpu().data.numpy()
    def metrics(self, loader):
        ys = [(y.cpu().data.numpy(),self.model(x).squeeze(-1).sigmoid().cpu().data.numpy()) for (x,y) in loader]
        y_gt,y_pred = zip(*ys)
        y_gt_all = np.concatenate(y_gt)
        y_pred_all = np.concatenate(y_pred)
        auroc = sk.metrics.roc_auc_score(y_gt_all, y_pred_all)
        average_precision = sk.metrics.average_precision_score(y_gt_all,y_pred_all)
        return pd.Series({'AUROC':auroc,'AP':average_precision,'NLL':self.evalAverageMetrics(loader,self.fresh_mll)})

    def logStuff(self, step, minibatch=None):
        super().logStuff(step, minibatch)

class PNCNNTrainer2Nounc(PNCNNTrainer2):
    def loss(self,minibatch):
        x, y = minibatch
        logit_pred = self.model(x).squeeze(-1)
        pos_weight = torch.tensor([self.dataloaders['train'].dataset.class_weights]).float().to(logit_pred.device)
        BCE_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logit_pred, y.float())
        return BCE_loss

    def metrics(self, loader):
        ys = [(y.cpu().data.numpy(),self.model(x).squeeze(-1).sigmoid().cpu().data.numpy()) for (x,y) in loader]
        y_gt,y_pred = zip(*ys)
        y_gt_all = np.concatenate(y_gt)
        y_pred_all = np.concatenate(y_pred)
        auroc = sk.metrics.roc_auc_score(y_gt_all, y_pred_all)
        average_precision = sk.metrics.average_precision_score(y_gt_all,y_pred_all)
        return pd.Series({'AUROC':auroc,'AP':average_precision})
