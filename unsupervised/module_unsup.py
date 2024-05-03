import os
import numpy as np
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

from unsupervised.model_unsup import Autoencoder

from pdb import set_trace

class VtdModule(LightningModule):
    def __init__(self, params):
        super().__init__()
        torch.manual_seed(params.seed)
        np.random.seed(params.seed)
        self.model = Autoencoder(params)
        self.criterion = nn.MSELoss()
        self.params = params
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        out = self(x)
        loss = self.criterion(out, x)
        self.log('train/loss', loss.item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        out = self(x)
        loss = self.criterion(out, x)
        self.log('val/loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

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