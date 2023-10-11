import os
import numpy as np
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

from src.model import CompClassModel
from utils.utils import ContrastiveLoss, DcfLoss
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from pdb import set_trace

class VtdModule(LightningModule):
    def __init__(self, params):
        super().__init__()
        torch.manual_seed(params.seed)
        np.random.seed(params.seed)
        self.model = CompClassModel(params)
        if params.class_loss=='xent':
            self.criterion = nn.NLLLoss(weight=torch.tensor([1, params.target_weight]))
        else:
            self.criterion = DcfLoss(fnr_weight=0.75, smax_weight=params.dsmax_mult)
        self.contrast_criterion = ContrastiveLoss(weight=params.target_weight)
        self.params = params
        self.save_hyperparameters()
        self.al_method = None
        self.val_fns = 0
        self.val_fps = 0
        self.val_ns = 0
        self.val_ps = 0
        self.test_fns = 0
        self.test_fps = 0
        self.test_ns = 0
        self.test_ps = 0
        self.test_scores = []
        self.val_scores = []
        self.test_labels = []
        self.val_labels = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x1,y1,x2,y2 = batch
        embed1, y_hat, _ = self(x1)
        if self.params.xent_weight != 1.0:
            embed2 = self(x2)[0]
            loss = self.params.xent_weight*self.criterion(y_hat,y1) \
                    + (1-self.params.xent_weight)*self.contrast_criterion(embed1,embed2,y1,y2)
        else:
            loss = self.criterion(y_hat,y1)
        self.log('train/loss', loss.item(), on_step=False, on_epoch=True)
        pred = torch.argmax(y_hat, dim=-1)
        acc = torch.mean(1.0*(pred==y1))
        self.log('train/acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y,_,_ = batch
        embeds,y_hat,logits = self(x)
        loss = self.criterion(y_hat,y)
        self.log('val/loss', loss.item(), on_step=False, on_epoch=True)
        pred = torch.argmax(y_hat, dim=-1)
        acc = torch.mean(1.0*(pred==y))
        self.log('val/acc', acc, on_step=False, on_epoch=True)

        self.val_fps += torch.sum(torch.logical_and(y==0, pred==1))
        self.val_fns += torch.sum(torch.logical_and(y==1, pred==0))
        self.val_ps += torch.sum(y==1)
        self.val_ns += torch.sum(y==0)
        self.val_scores.append(torch.exp(y_hat[:,1]))
        self.val_labels.append(y)

    def on_validation_epoch_end(self):
        fnr = self.val_fns/self.val_ps if self.val_ps > 0 else 0.0
        fpr = self.val_fps/self.val_ns if self.val_ns > 0 else 0.0
        dcf = 0.25*fpr + 0.75*fnr
        acc = 1-(self.val_fns+self.val_fps)/(self.val_ps+self.val_ns)

        # scores = torch.cat(self.val_scores, 0).detach().cpu().numpy()
        # labels = torch.cat(self.val_labels, 0).detach().cpu().numpy()
        # precisions, recalls, _ = precision_recall_curve(labels, scores)
        # sorted_idxs = np.argsort(recalls)
        # recalls = recalls[sorted_idxs]
        # precisions = precisions[sorted_idxs]
        # fprs, tprs, thresholds = roc_curve(labels, scores)
        # auprc = auc(recalls, precisions)

        self.log('val/fnr', fnr)
        self.log('val/fpr', fpr)
        self.log('val/dcf', dcf, prog_bar=True)
        self.log('val/acc', acc)
        # self.log('val/auprc', auprc)
        self.log('val/ps', float(self.val_ps))
        self.log('val/fps', float(self.val_fps))
        self.log('val/ns', float(self.val_ns))
        self.log('val/fns', float(self.val_fns))

        self.val_fns = 0
        self.val_fps = 0
        self.val_ps = 0
        self.val_ns = 0
        self.val_scores = []
        self.val_labels = []

    def test_step(self, batch, batch_idx):
        x,y,_,_ = batch
        y_hat = self(x)[1]
        loss = self.criterion(y_hat,y)
        self.log('test/loss', loss.item(), on_step=False, on_epoch=True)
        pred = torch.argmax(y_hat, dim=-1)
        acc = torch.mean(1.0*(pred==y))
        self.log('test/acc', acc, on_step=False, on_epoch=True)

        self.test_fps += torch.sum(torch.logical_and(y==0, pred==1))
        self.test_fns += torch.sum(torch.logical_and(y==1, pred==0))
        self.test_ps += torch.sum(y==1)
        self.test_ns += torch.sum(y==0)
        self.test_scores.append(torch.exp(y_hat[:,1]))
        self.test_labels.append(y)

    def on_test_epoch_end(self):
        fnr = self.test_fns/self.test_ps if self.test_ps > 0 else 0.0
        fpr = self.test_fps/self.test_ns if self.test_ns > 0 else 0.0
        dcf = 0.25*fpr + 0.75*fnr
        acc = 1-(self.test_fns+self.test_fps)/(self.test_ps+self.test_ns)

        # scores = torch.cat(self.test_scores, 0).detach().cpu().numpy()
        # labels = torch.cat(self.test_labels, 0).detach().cpu().numpy()
        # precisions, recalls, _ = precision_recall_curve(labels, scores)
        # sorted_idxs = np.argsort(recalls)
        # recalls = recalls[sorted_idxs]
        # precisions = precisions[sorted_idxs]
        # fprs, tprs, thresholds = roc_curve(labels, scores)
        # auprc = auc(recalls, precisions)

        self.log('test/fnr', fnr)
        self.log('test/fpr', fpr)
        self.log('test/dcf', dcf)
        self.log('test/acc', acc)
        # self.log('test/auprc', auprc)
        self.log('test/ps', float(self.test_ps))
        self.log('test/fps', float(self.test_fps))
        self.log('test/ns', float(self.test_ns))
        self.log('test/fns', float(self.test_fns))

        self.test_fns = 0
        self.test_fps = 0
        self.test_ps = 0
        self.test_ns = 0
        self.test_scores = []
        self.test_labels = []

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.params.lr,
                                weight_decay=self.params.wd)
        if self.params.schedule_lr:
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=self.params.mode,
                                                             patience=self.params.lrs_patience, 
                                                             verbose=True, threshold=0)
            return {
                'optimizer': opt,
                'lr_scheduler': {
                    'scheduler': sch,
                    'monitor': self.params.monitor
                }
            }
        else:
            return opt