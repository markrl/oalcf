import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import numpy as np
from statistics import mean
from copy import deepcopy

from pdb import set_trace

class ImlDataModule(LightningDataModule):
    def __init__(self, params):
        self.params = params
        super().__init__()
        # Form dataset
        if 'VTD' in params.feat_root:
            self.ds = BaseVtdData(params)
            self.task = 'vtd'
        elif 'LID' in params.feat_root:
            self.ds = BaseLidData(params)
            self.task = 'lid'
        self.data_train = ImlData(params, self.ds)
        self.data_test = ImlData(params, self.ds)
        self.current_batch = -1
        self.n_batches = int(len(self.ds)/params.samples_per_batch)
        self.drop_last = False
        self.train_active_order = {}
        self.forget_n_batches = params.forget_n_batches

    def setup(self, stage):
        return

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=False,
            drop_last=self.drop_last,
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

    def idx_loader(self, idxs):
        ds = ImlData(self.params, self.ds)
        ds.activate_samples(idxs)
        return DataLoader(
            dataset=ds,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=False,
        )

    def label_boot(self):
        if self.params.bootstrap==0: # No bootstrap
            return
        elif self.params.boot_in: # Construct bootstrap corpus directly from evaluation data (in-domain)
            n_nontarget = int(self.params.bootstrap/2)
            n_target = self.params.bootstrap-n_nontarget
            target_idxs, nontarget_idxs = [], []
            ii = 0
            while ii<len(self.ds) and (len(target_idxs)<n_target or len(nontarget_idxs)<n_nontarget):
                label = self.ds.get_label(ii)
                if label==0 and len(nontarget_idxs)<n_nontarget:
                    nontarget_idxs.append(ii)
                elif label==1 and len(target_idxs)<n_target:
                    target_idxs.append(ii)
                ii += 1
            idxs = nontarget_idxs + target_idxs
        else: # Construct bootstrap corpus from training data (out-of-domain)
            idxs = self.ds.load_external_bootstrap()
            self.data_train.reset_idxs()
            self.data_test.reset_idxs()
        self.data_train.activate_samples(idxs)
        self.train_active_order[self.current_batch] = idxs

    def get_class_counts(self):
        labels = torch.FloatTensor([self.ds.get_label(ii) for ii in self.data_train.active_idxs])
        n_target = torch.sum(labels)
        n_nontarget = len(labels)-n_target
        return n_target,n_nontarget
    
    def get_class_balance(self):
        n_target,n_nontarget = self.get_class_counts()
        p_target = n_target/(n_target+n_nontarget)
        p_nontarget = n_nontarget/(n_target+n_nontarget)
        return p_target,p_nontarget

    def transfer_samples(self, idxs):
        # Convert given inidices to base indices. This is the only function where this is necessary.
        idxs = [self.data_test.active_idxs[ii] for ii in idxs]
        # Activate train, deactivate test
        self.data_train.activate_samples(idxs)
        self.data_test.deactivate_samples(idxs)
        self.train_active_order[self.current_batch] = idxs

    def forget_samples(self):
        if self.forget_n_batches is not None:
            batch_num = self.current_batch - self.forget_n_batches
            orig_keys = list(self.train_active_order.keys())
            for kk in orig_keys:
                if kk <= batch_num:
                    idxs = self.train_active_order.pop(kk)
                    self.data_train.deactivate_samples(idxs) 

    def next_batch(self):
        if self.current_batch>=self.n_batches-1:
            self.current_batch += 1
            return
        # Activate the next batch
        self.current_batch += 1
        self.data_test.deactivate_all()
        self.data_test.activate_samples([nn for nn in range((self.current_batch+self.ds.n_boot_batches)*self.params.samples_per_batch, 
                                                            (self.current_batch+self.ds.n_boot_batches+1)*self.params.samples_per_batch)])

        # Check for bootstrap samples
        self.data_test.deactivate_samples(self.data_train.active_idxs)

    def unlabeled_len(self):
        return len(self.data_test)

    def __len__(self):
        return len(self.data_train)

    def save_active_files(self, path):
        file_list = [os.path.basename(self.ds.feat_files[int(ii/self.params.samples_per_batch)])[:-4].replace('_{}', '')
                for ii in self.data_train.active_idxs]
        sample_list = [str(ii%self.params.samples_per_batch) for ii in self.data_train.active_idxs]
        lines = [f'{ff},{ss}' for ff,ss in zip(file_list, sample_list)]
        content = '\n'.join(lines) + '\n'
        with open(path, 'w') as f:
            f.write(content)

    def get_current_session_name(self):
        if len(self.data_test)==0:
            return 'NO BATCH LOADED'
        elif self.task=='vtd':
            file_idx = int(self.data_test.active_idxs[0] / self.params.samples_per_batch)
            se_name = self.ds.feat_files[file_idx].replace('_{}', '').replace('.npy', '')
            return se_name.upper()
        elif self.task=='lid':
            return f'SESSION {self.current_batch}'

    def get_train_labels(self):
        if len(self.data_train)==0:
            return None
        labels = [self.data_train.get_label(ii) for ii in np.arange(len(self.data_train))]
        return torch.LongTensor(labels)

    def get_test_labels(self):
        if len(self.data_test)==0:
            return None
        labels = [self.data_test.get_label(ii) for ii in np.arange(len(self.data_test))]
        return torch.LongTensor(labels)
    

class BaseVtdData(Dataset):
    def __init__(self, 
                 params,
                 ):
        super().__init__()
        self.params = params
        self.n_boot_batches = 0
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
        feat = np.concatenate([np.load(os.path.join(rr,self.feat_files[file_idx].format(nn)))[start_idx:stop_idx] 
                            for rr,nn in zip(self.feat_roots, self.feat_names)], axis=1)
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

    def get_class_balance(self):
        labels = []
        for ii in range(len(self)):
            labels.append(self.get_label(ii))
        p_target = np.mean(labels)
        p_nontarget = 1-p_target
        print(f'{p_target*100:.2f}% target')
        print(f'{p_nontarget*100:.2f}% nontarget')

    def load_external_bootstrap(self):
        train_params = deepcopy(self.params)
        rm = self.params.env_name.split('_')[0]
        if rm in ['rm1', 'rm2', 'rm3', 'rm4']:
            train_params.env_name = 'apartment_mc19'
        else:
            train_params.env_name = 'rm1_mc20'
        train_ds = BaseVtdData(train_params)
        n_nontarget = int(self.params.bootstrap/2)
        n_target = self.params.bootstrap-n_nontarget
        target_idxs, nontarget_idxs = [], []
        ii = 0
        while ii<len(train_ds) and (len(target_idxs)<n_target or len(nontarget_idxs)<n_nontarget):
            label = train_ds.get_label(ii)
            if label==0 and len(nontarget_idxs)<n_nontarget:
                nontarget_idxs.append(ii)
            elif label==1 and len(target_idxs)<n_target:
                target_idxs.append(ii)
            ii += 1
        idxs = nontarget_idxs + target_idxs
        self.n_boot_batches = int(np.max(idxs)/self.params.samples_per_batch) + 1
        self.label_files = train_ds.label_files[:self.n_boot_batches] + self.label_files
        self.feat_files = train_ds.feat_files[:self.n_boot_batches] + self.feat_files
        return idxs


class BaseLidData(Dataset):
    def __init__(self, 
                 params,
                 ):
        super().__init__()
        self.params = params
        self.n_boot_batches = 0
        self.env = params.env_name
        self.feat_roots = params.feat_root.split(',')
        self.feat_roots = [fr[:-1] if fr[-1]=='/' else fr for fr in self.feat_roots]
        self.feat_names = [rr.split('/')[-1] for rr in self.feat_roots]
        self.feat_files = []
        self.labels = []
        prefix = os.path.join(self.feat_roots[0], '..')
        order_file = os.path.join(prefix, 'orders', f'{params.order_file}_order_{self.env}')
        with open(order_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                label = line.split('_')[2]
                if label == params.lid_target:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
                self.feat_files.append(os.path.join(prefix, '{}', self.env, line+'.npy'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.get_label(index)
        feat = self.get_sample_with_context(index)
        return torch.from_numpy(feat).float(), label

    def get_label(self, index):
        return int(self.labels[index])

    def get_class_balance(self):
        labels = []
        for ii in range(len(self)):
            labels.append(self.get_label(ii))
        p_target = np.mean(labels)
        p_nontarget = 1-p_target
        print(f'{p_target*100:.2f}% target')
        print(f'{p_nontarget*100:.2f}% nontarget')

    def get_sample_with_context(self, index):
        idx_in_batch = index % self.params.samples_per_batch
        max_idx = len(self)-1
        feat = []
        for ii in range(index-self.params.context, index+self.params.context+1):
            idx = ii % max_idx
            feat.append(np.concatenate([np.load(self.feat_files[idx].format(nn)) for nn in self.feat_names]))
        feat = np.stack(feat, axis=0)
        if idx_in_batch<self.params.context:
            n_zeros = self.params.context-idx_in_batch
            feat[:n_zeros] = 0
        if idx_in_batch+self.params.context>=self.params.samples_per_batch:
            n_zeros = self.params.context+idx_in_batch-self.params.samples_per_batch+1
            feat[-n_zeros:] = 0
        return feat.reshape(-1)

    def load_external_bootstrap(self):
        train_params = deepcopy(self.params)
        train_params.env_name = 'train'
        train_ds = BaseLidData(train_params)
        n_nontarget = int(self.params.bootstrap/2)
        n_target = self.params.bootstrap-n_nontarget
        target_idxs, nontarget_idxs = [], []
        ii = 0
        while ii<len(train_ds) and (len(target_idxs)<n_target or len(nontarget_idxs)<n_nontarget):
            label = train_ds.get_label(ii)
            if label==0 and len(nontarget_idxs)<n_nontarget:
                nontarget_idxs.append(ii)
            elif label==1 and len(target_idxs)<n_target:
                target_idxs.append(ii)
            ii += 1
        idxs = nontarget_idxs + target_idxs
        self.n_boot_batches = int(np.max(idxs)/self.params.samples_per_batch) + 1
        self.labels = train_ds.labels[:self.n_boot_batches*self.params.samples_per_batch] + self.labels
        self.feat_files = train_ds.feat_files[:self.n_boot_batches*self.params.samples_per_batch] + self.feat_files
        return idxs


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
        idx2 = self.active_idxs[np.random.randint(low=0, high=len(self))]
        feat2, label2 = self.base_ds[idx2]
        return feat1, label1, feat2, label2
    
    def get_label(self, index):
        index = self.active_idxs[index]
        return self.base_ds.get_label(index)

    def cat_data(self):
        feats = []
        for idx in self.active_idxs:
            feats.append(self.base_ds[idx][0])
        return torch.stack(feats, dim=0)

    def reset_idxs(self):
        self.active_idxs = []
        self.inactive_idxs = [nn for nn in range(len(self.base_ds))]

if __name__=='__main__':
    from params import get_params
    params = get_params()
    data_module = ImlDataModule(params)
    data_module.ds[len(data_module.ds)-1]
    data_module.label_boot()
    data_module.next_batch()
    print(data_module.data_test[4])
    print(data_module.data_train[0])
    data_module.transfer_samples([2,4])
    data_module.next_batch()
    data_module.transfer_samples([3,5])
    print(data_module.get_current_session_name())
    data_module.ds.get_class_balance()
