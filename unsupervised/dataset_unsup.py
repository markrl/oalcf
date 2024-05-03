import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import numpy as np
from statistics import mean
from copy import deepcopy

from pdb import set_trace

class VtdDataModule(LightningDataModule):
    def __init__(self, params):
        self.params = params
        super().__init__()
        # Form dataset
        self.data_train = VtdData(params)

    def setup(self, stage):
        return

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=False,
        )

    def __len__(self):
        return len(self.data_train)


class VtdData(Dataset):
    def __init__(self, 
                 params,
                 ):
        super().__init__()
        self.params = params
        rm, mc = params.env_name.split('_')

        self.label_files = glob.glob(os.path.join(params.ann_root, '*.npy'))
        self.label_files = [ff for ff in self.label_files if rm in ff]
        self.label_files.sort()
        
        self.feat_roots = params.feat_root.split(',')
        self.feat_names = [os.listdir(rr)[0].split('_')[3] for rr in self.feat_roots]
        self.feat_files = []
        bad_files = []
        for ff in self.label_files:
            rm,se,splt = os.path.basename(ff).split('_')
            feat_file = '_'.join((rm, se, mc, '{}', splt))
            self.feat_files.append(feat_file)
            if not np.all([os.path.exists(os.path.join(rr, feat_file).format(nn)) for rr,nn in zip(self.feat_roots, self.feat_names)]):
                bad_files.append((ff, feat_file))
        for lf,ff in bad_files:
            self.label_files.remove(lf)
            self.feat_files.remove(ff)

    def __len__(self):
        return len(self.feat_files)*self.params.samples_per_batch

    def __getitem__(self, index):
        feat = self.get_sample_with_context(index)
        return torch.from_numpy(feat).float()

    def get_sample_with_context(self, index):
        file_idx = int(index / self.params.samples_per_batch)
        sample_idx = index % self.params.samples_per_batch
        start_idx = np.maximum(0, sample_idx-self.params.context)
        stop_idx = np.minimum(self.params.samples_per_batch, sample_idx+self.params.context+1)
        feat = np.concatenate([np.load(os.path.join(rr,self.feat_files[file_idx].format(nn)))[start_idx:stop_idx] 
                            for rr,nn in zip(self.feat_roots, self.feat_names)], axis=1)
        if sample_idx<self.params.context:
            n_zeros = self.params.context-sample_idx
            feat = np.concatenate([np.zeros((n_zeros, feat.shape[1])), feat], axis=0)
        if sample_idx+self.params.context>=self.params.samples_per_batch:
            n_zeros = self.params.context+sample_idx-self.params.samples_per_batch+1
            feat = np.concatenate([feat, np.zeros((n_zeros, feat.shape[1]))], axis=0)
        return feat.reshape(-1)