import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
import numpy as np
from statistics import mean
from copy import deepcopy

from pdb import set_trace

class VtdImlDataModule(LightningDataModule):
    def __init__(self, params):
        self.params = params
        super().__init__()
        # Form dataset
        self.ds = BaseImlData(params)
        self.data_train = ImlData(params, self.ds)
        self.data_test = ImlData(params, self.ds)
        self.current_batch = 0
        self.n_batches = int(len(self.ds)/params.samples_per_batch)

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

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=False,
        )

    def label_boot(self):
        if self.params.bootstrap==0:
            return
        n_nontarget = int(self.params.bootstrap/2)
        n_target = self.params.bootstrap-n_nontarget
        target_idxs, nontarget_idxs = [], []
        ii = 0
        while len(target_idxs)<n_target or len(nontarget_idxs)<n_nontarget:
            label = self.ds.get_label(ii)
            if label==0 and len(nontarget_idxs)<n_nontarget:
                nontarget_idxs.append(ii)
            elif label==1 and len(target_idxs)<n_target:
                target_idxs.append(ii)
            ii += 1
        idxs = nontarget_idxs + target_idxs
        self.data_train.activate_samples(idxs)
    
    def get_class_balance(self):
        labels = torch.FloatTensor([self.ds.get_label(ii) for ii in self.data_train.active_idxs])
        p_target = torch.mean(labels)
        p_nontarget = 1 - p_target
        return p_target,p_nontarget

    def transfer_samples(self, idxs):
        # Convert inidices to base indices
        idxs = [self.data_test.active_idxs[ii] for ii in idxs]
        # Activate train, deactivate test
        self.data_train.activate_samples(idxs)
        self.data_test.deactivate_samples(idxs)

    def next_batch(self):
        if self.current_batch>=self.n_batches:
            self.current_batch += 1
            return
        # Activate the next batch
        self.data_test.deactivate_all()
        self.data_test.activate_samples([nn for nn in range(self.current_batch*self.params.samples_per_batch, 
                                                            (self.current_batch+1)*self.params.samples_per_batch)])
        self.current_batch += 1

        # Check for bootstrap samples
        self.data_test.deactivate_samples(self.data_train.active_idxs)

    def unlabeled_len(self):
        return len(self.data_test)

    def __len__(self):
        return len(self.data_train)

    def save_active_files(self, path):
        file_list = [os.path.basename(self.ds.feat_files[int(ii/self.params.samples_per_batch)])[:-3]
                for ii in self.data_train.active_idxs]
        sample_list = [str(ii%self.params.samples_per_batch) for ii in self.data_train.active_idxs]
        lines = [f'{ff} {ss}' for ff,ss in zip(file_list, sample_list)]
        content = '\n'.join(lines) + '\n'
        with open(path, 'w') as f:
            f.write(content)

    def get_current_session_name(self):
        file_idx = int(self.data_test.active_idxs[0] / self.params.samples_per_batch)
        se_name = self.ds.feat_files[file_idx].replace('_wavlm', '').replace('.npy', '')
        return se_name.upper()
    

class BaseImlData(Dataset):
    def __init__(self, 
                 params,
                 ):
        super().__init__()
        self.params = params
        rm, mc = params.env_name.split('_')
        self.feat_roots = params.feat_root.split(',')
        self.feat_files = os.listdir(os.path.join(self.feat_roots[0]))
        self.feat_files = [ff for ff in self.feat_files if (rm in ff and mc in ff)]
        self.feat_files.sort()

        self.label_files = glob.glob(os.path.join(params.ann_root, '*.npy'))
        self.label_files = [ff for ff in self.label_files if rm in ff]
        self.label_files.sort()

    def __len__(self):
        return len(self.label_files)*self.params.samples_per_batch

    def __getitem__(self, index):
        label = self.get_label(index)
        feat = self.get_sample_with_context(index)
        return torch.from_numpy(feat).float(), label

    def get_sample_with_context(self, index):
        file_idx = int(index / self.params.samples_per_batch)
        sample_idx = index % self.params.samples_per_batch
        start_idx = np.maximum(0, sample_idx-self.params.context)
        stop_idx = np.minimum(self.params.samples_per_batch, sample_idx+self.params.context+1)
        feat = np.concatenate([np.load(os.path.join(rr,self.feat_files[file_idx]))[start_idx:stop_idx] 
                            for rr in self.feat_roots], axis=1)
        if sample_idx<self.params.context:
            n_zeros = self.params.context-sample_idx
            feat = np.concatenate([np.zeros((n_zeros, feat.shape[1])), feat], axis=0)
        if sample_idx+self.params.context>=self.params.samples_per_batch:
            n_zeros = self.params.context+sample_idx-self.params.samples_per_batch+1
            feat = np.concatenate([feat, np.zeros((n_zeros, feat.shape[1]))], axis=0)
        return feat.reshape(-1)

    def get_label(self, index):
        file_idx = int(index / self.params.samples_per_batch)
        sample_idx = index % self.params.samples_per_batch
        return int(np.load(self.label_files[file_idx])[sample_idx])


class ImlData(Dataset):
    def __init__(self, params, base_ds):
        super().__init__()
        self.params = params
        self.base_ds = base_ds

        self.active_idxs = []
        self.inactive_idxs = [nn for nn in range(len(self.base_ds))]

    def __len__(self):
        return len(self.active_idxs)

    def inactive_len(self):
        return len(self.inactive_idxs)

    def empty_check(self):
        return self.inactive_len()==0

    def activate_samples(self, idxs):
        # Here, idxs refers to the numbers stored in the active_idx list, not the indices of that list
        assert not self.empty_check()

        if type(idxs) == int:
            idxs = [idxs]

        idxs.sort()
        idxs = [ii for ii in idxs if ii not in self.active_idxs]
        self.active_idxs += idxs
        idxs.reverse()
        for idx in idxs:
            self.inactive_idxs.remove(idx)

        self.active_idxs.sort()
        self.inactive_idxs.sort()

        assert len(self.active_idxs) + len(self.inactive_idxs) == len(self.base_ds)
    
    def deactivate_samples(self, idxs):
        # Here, idxs refers to the numbers stored in the inactive_idx list, not the indices of that list
        assert not self.empty_check()

        if type(idxs) == int:
            idxs = [idxs]

        idxs.sort()
        idxs = [ii for ii in idxs if ii not in self.inactive_idxs]
        self.inactive_idxs += idxs
        idxs.reverse()
        for idx in idxs:
            self.active_idxs.remove(idx)

        self.active_idxs.sort()
        self.inactive_idxs.sort()

        assert len(self.active_idxs) + len(self.inactive_idxs) == len(self.base_ds)

    def deactivate_all(self):
        self.inactive_idxs += self.active_idxs
        self.active_idxs = []

    def activate_all(self):
        self.active_idxs += self.inactive_idxs
        self.inactive_idx = []

    def __getitem__(self, index):
        index = self.active_idxs[index]
        feat1, label1 = self.base_ds[index]
        idx2 = self.active_idxs[torch.randint(0,len(self),(1,))[0]]
        feat2, label2 = self.base_ds[idx2]
        return feat1, label1, feat2, label2

    def cat_data(self):
        feats = []
        for idx in self.active_idxs:
            feats.append(self.base_ds[idx][0])
        return torch.stack(feats, dim=0)

if __name__=='__main__':
    from params import get_params
    params = get_params()
    data_module = VtdImlDataModule(params)
    data_module.label_boot()
    data_module.next_batch()
    data_module.transfer_samples([2,4])
    data_module.next_batch()
    data_module.transfer_samples([3,5])
    print(data_module.get_current_session_name())
    # data_module.save_active_files('/home/marklind/test.txt')