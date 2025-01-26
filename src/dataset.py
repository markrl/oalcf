import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
import numpy as np
from sklearn.cluster import KMeans
from statistics import mean
from copy import deepcopy

from pdb import set_trace

class ImlDataModule(LightningDataModule):
    def __init__(self, params):
        self.params = params
        super().__init__()
        # Form dataset
        if 'VTD' in params.feat_root.upper():
            self.ds = BaseVtdData(params)
            self.task = 'vtd'
        elif 'LID' in params.feat_root.upper():
            self.ds = BaseLidData(params)
            self.task = 'lid'
        self.data_train = ImlData(params, True, self.ds)
        self.data_test = ImlData(params, False, self.ds)
        self.current_batch = -1
        self.n_batches = int(len(self.ds)/params.samples_per_batch)
        self.drop_last = False
        self.train_active_order = {}
        self.forget_n_batches = params.forget_n_batches
        self.use_gpu = params.gpus > 0 and torch.cuda.is_available()

    def setup(self, stage):
        return

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=self.params.shuffle_train,
            drop_last=self.drop_last,
            # pin_memory=True,
        )

    def val_dataloader(self):
        data_val = ImlData(self.params, False, train_ds=self.data_train)
        return DataLoader(
            dataset=data_val,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=False,
            # pin_memory=True,
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
        self.data_train.limit_data(self.data_test)
        if self.params.pair_type=='rand' and self.params.contrast_loss=='triplet':
            self.data_train.update_class_idx_lists()

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

    def transfer_samples(self, idxs, model=None):
        # Convert given inidices to base indices. This is the only function where this is necessary.
        idxs = [self.data_test.active_idxs[ii] for ii in idxs]
        # No need to add samples to the adaptation pool twice
        active_order = []
        keys = list(self.train_active_order.keys())
        keys.sort()
        for kk in keys:
            active_order += self.train_active_order[kk]
        idxs = [ii for ii in idxs if ii not in active_order]
        active_order += idxs
        if self.params.memory_buffer is None:
            # Activate train, deactivate test
            self.data_train.activate_samples(idxs)
            if self.current_batch in self.train_active_order.keys():
                self.train_active_order[self.current_batch] += idxs
            else:
                self.train_active_order[self.current_batch] = idxs
        elif self.params.memory_buffer == 'ring':
            # FIFO
            self.data_train.activate_samples(idxs)
            if self.current_batch in self.train_active_order.keys():
                self.train_active_order[self.current_batch] += idxs
            else:
                self.train_active_order[self.current_batch] = idxs
            if len(self.data_train.active_idxs) > self.params.buffer_cap:
                rm_idxs = active_order[:-self.params.buffer_cap]
                self.data_train.deactivate_samples(rm_idxs)
                for kk in self.train_active_order:
                    for ii in rm_idxs:
                        if ii in self.train_active_order[kk]:
                            self.train_active_order[kk].remove(ii)
        elif self.params.memory_buffer == 'class':
            # FIFO, keep class balance
            self.data_train.activate_samples(idxs)
            if self.current_batch in self.train_active_order.keys():
                self.train_active_order[self.current_batch] += idxs
            else:
                self.train_active_order[self.current_batch] = idxs
            if len(self.data_train.active_idxs) > self.params.buffer_cap:
                labels = np.array([self.data_train.get_label(ii) for ii in range(len(self.data_train))])
                active_order_target = np.array(active_order)[labels==1]
                active_order_nontarget = np.array(active_order)[labels==0]
                if len(active_order_target) > self.params.buffer_cap/2 and len(active_order_nontarget) > self.params.buffer_cap/2:
                    rm_idxs = active_order_target[int(self.params.buffer_cap/2-len(active_order_target)):]
                    rm_idxs = np.append(rm_idxs, active_order_nontarget[int(self.params.buffer_cap/2-len(active_order_nontarget)):])
                elif len(active_order_nontarget) > self.params.buffer_cap/2:
                    rm_idxs = active_order_nontarget[self.params.buffer_cap-len(active_order):]
                elif len(active_order_target) > self.params.buffer_cap/2:
                    rm_idxs = active_order_target[self.params.buffer_cap-len(active_order):]
                self.data_train.deactivate_samples(rm_idxs)
                for kk in self.train_active_order:
                    for ii in rm_idxs:
                        if ii in self.train_active_order[kk]:
                            self.train_active_order[kk].remove(ii)
        elif self.params.memory_buffer == 'confidence':
            # Not FIFO, choose the least confident samples
            self.data_train.activate_samples(idxs)
            if self.current_batch in self.train_active_order.keys():
                self.train_active_order[self.current_batch] += idxs
            else:
                self.train_active_order[self.current_batch] = idxs
            if len(self.data_train.active_idxs) > self.params.buffer_cap:
                confidence_scores = self.get_confidence(model)
                sorted_idxs = np.argsort(confidence_scores)
                rm_idxs = np.array(self.data_train.active_idxs)[sorted_idxs[self.params.buffer_cap-len(self.data_train.active_idxs):]]
                self.data_train.deactivate_samples(rm_idxs)
                for kk in self.train_active_order:
                    for ii in rm_idxs:
                        if ii in self.train_active_order[kk]:
                            self.train_active_order[kk].remove(ii)
        else:
            print('Not a valid memory buffer')
            exit()
        self.data_train.limit_data(self.data_test)
        if self.params.pair_type=='rand' and self.params.contrast_loss=='triplet':
            self.data_train.update_class_idx_lists()

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
    
    def get_confidence(self, model):
        model.eval()
        loader = self.train_dataloader()
        with torch.no_grad():
            if self.use_gpu:
                model = model.cuda()
                logits = [model(batch[0].cuda())[-1] for batch in loader]
            else:
                logits = [model(batch[0])[-1] for batch in loader]
        logits = torch.cat(logits, dim=0).cpu()
        scores, _ = torch.max(F.softmax(logits, dim=-1), dim=-1)
        return scores.numpy()
    

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

        self.feat_files *= params.repeat_dataset
        self.label_files *= params.repeat_dataset

        if params.reverse_order:
            self.label_files.reverse()
            self.feat_files.reverse()

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

        self.feat_files *= params.repeat_dataset
        self.labels *= params.repeat_dataset

        if params.reverse_order:
            self.feat_files.reverse()
            self.labels.reverse()

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
    def __init__(self, params, training, base_ds=None, train_ds=None):
        super().__init__()
        self.params = params
        if train_ds is not None:
            self.base_ds = train_ds.base_ds
            self.active_idxs = train_ds.active_idxs
            self.inactive_idxs = train_ds.inactive_idxs
        else:
            self.base_ds = base_ds
            self.active_idxs = []
            self.inactive_idxs = [nn for nn in range(len(self.base_ds))]
        self.training = training
        if training:
            if params.pair_type=='offset':
                self.offset = 1
            elif params.pair_type=='neighbors':
                self.close_neighbors = True
                self.extremes = {}
            
            if params.limit_train_size is not None:
                self.limit_train_size = params.limit_train_size
                self.limited_idxs = []
                self.n_clusters = 2
                self.cluster = KMeans(n_clusters=self.n_clusters, n_init='auto')
            else:
                self.limit_train_size = None
        else:
            self.limit_train_size = None
        if params.pair_type=='rand' and params.contrast_loss=='triplet':
            self.target_idx_list = []
            self.nontarget_idx_list = []

    def __len__(self):
        return len(self.active_idxs) if self.limit_train_size is None else len(self.limited_idxs)

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
        idx_converter = self.active_idxs if self.limit_train_size is None else self.limited_idxs
        if self.training:
            anchor_idx = idx_converter[index]
            anchor, anchor_label = self.base_ds[anchor_idx]
            if self.params.contrast_loss=='triplet':
                if self.params.pair_type=='rand':
                    if anchor_label==1:
                        pos_idx = self.target_idx_list[np.random.randint(low=0, high=len(self.target_idx_list))]
                        neg_idx = self.nontarget_idx_list[np.random.randint(low=0, high=len(self.nontarget_idx_list))]
                    else:
                        pos_idx = self.nontarget_idx_list[np.random.randint(low=0, high=len(self.nontarget_idx_list))]
                        neg_idx = self.target_idx_list[np.random.randint(low=0, high=len(self.target_idx_list))]
                elif self.params.pair_type=='neighbors':
                    pos_idx = idx_converter[self.extremes[index][False]]
                    neg_idx = idx_converter[self.extremes[index][True]]
                pos, pos_label = self.base_ds[pos_idx]
                neg, neg_label = self.base_ds[neg_idx]
                return anchor, anchor_label, pos, pos_label, neg, neg_label
            else:
                idx1 = idx_converter[index]
                feat1, label1 = self.base_ds[idx1]
                if self.params.pair_type=='rand':
                    idx2 = idx_converter[np.random.randint(low=0, high=len(self))]
                elif self.params.pair_type=='offset':
                    idx2 = idx_converter[self.choose_pair_idx(index)]
                elif self.params.pair_type=='neighbors':
                    idx2 = idx_converter[self.extremes[index][self.close_neighbors]]
                feat2, label2 = self.base_ds[idx2]
                return feat1, label1, feat2, label2
        else:
            idx1 = idx_converter[index]
            feat1, label1 = self.base_ds[idx1]
            data_idx = idx1 - self.base_ds.n_boot_batches*self.params.samples_per_batch
            return feat1, label1, data_idx
    
    def get_label(self, index):
        index = self.active_idxs[index]
        return self.base_ds.get_label(index)

    def update_class_idx_lists(self):
        self.target_idx_list = []
        self.nontarget_idx_list = []
        for idx in self.active_idxs:
            label = self.base_ds.get_label(idx)
            if label==1:
                self.target_idx_list.append(idx)
            else:
                self.nontarget_idx_list.append(idx)
    
    def limit_data(self, comp_data):
        if self.limit_train_size is not None and len(self.active_idxs) > self.limit_train_size and len(comp_data) >= self.n_clusters:
            # Cluster comparison data
            comp_data = np.array([comp_data[nn][0].numpy() for nn in range(len(comp_data))])
            train_data = np.array([self[nn][0].numpy() for nn in range(len(self))])
            self.cluster.fit(comp_data)
            centers = self.cluster.cluster_centers_
            # Choose the samples that are most similar to the comparison data
            dists = [np.linalg.norm(center-train_data, axis=-1) for center in centers]
            mean_dists = np.mean(dists, axis=0)
            self.limited_idxs = [self.active_idxs[ii] for ii in np.argsort(mean_dists)][:self.limit_train_size]
        else:
            self.limited_idxs = self.active_idxs
    
    def choose_pair_idx(self, index):
        pair_index = (index + self.offset) % len(self)
        return pair_index
    
    def inc_offset(self):
        self.offset = 1 if self.offset==len(self)-1 else self.offset+1

    def compute_extremes(self, dists):
        self.extremes = {}
        labels = np.array([self.get_label(ii) for ii in range(len(self))])
        for ii in range(len(self)):
            self.extremes[ii] = {}
            sorted_idxs = np.argsort(dists[ii])
            same_labels = labels==labels[ii]
            if np.sum(same_labels)==0:
                # Both are close but different class
                self.extremes[ii][True] = sorted_idxs[np.where(1-same_labels[sorted_idxs])[0][0]]
                self.extremes[ii][False] = sorted_idxs[np.where(1-same_labels[sorted_idxs])[0][0]]
            elif np.sum(1-same_labels)==0:
                # Both are far but same class
                self.extremes[ii][True] = np.flipud(sorted_idxs)[np.where(same_labels[np.flipud(sorted_idxs)])[0][0]]
                self.extremes[ii][False] = np.flipud(sorted_idxs)[np.where(same_labels[np.flipud(sorted_idxs)])[0][0]]
            else:
                # Far but same class
                self.extremes[ii][False] = np.flipud(sorted_idxs)[np.where(same_labels[np.flipud(sorted_idxs)])[0][0]]
                # Close but different class
                self.extremes[ii][True] = sorted_idxs[np.where(1-same_labels[sorted_idxs])[0][0]]
                # Semi-hard negative
                # pos_dist = dists[ii,self.extremes[ii][False]]
                # if np.sum(np.logical_and(1-same_labels[sorted_idxs], dists[ii]>pos_dist))>0:
                #     self.extremes[ii][True] = sorted_idxs[np.where(np.logical_and(1-same_labels[sorted_idxs], dists[ii]>pos_dist))[0][0]]
                # else:
                #     self.extremes[ii][True] = np.flipud(sorted_idxs)[np.where(1-same_labels[np.flipud(sorted_idxs)])[0][0]]

    def cat_data(self):
        feats = []
        for idx in self.active_idxs:
            feats.append(self.base_ds[idx][0])
        return torch.stack(feats, dim=0)

    def reset_idxs(self):
        self.active_idxs = []
        self.inactive_idxs = [nn for nn in range(len(self.base_ds))]


class IncOffsetCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        trainer.datamodule.data_train.inc_offset()


class ToggleNeighborsCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.use_gpu = trainer.model.params.gpus > 0 and torch.cuda.is_available()
        self.ensemble = trainer.model.params.ensemble
        if trainer.datamodule.data_train.close_neighbors:
            # Get embeddings
            embeds = self.extract_embeds(trainer.datamodule, trainer.model)
            # Get distances
            dists = self.get_dists(embeds)
            # Compute extremes
            trainer.datamodule.data_train.compute_extremes(dists.numpy())
        trainer.datamodule.data_train.close_neighbors = not trainer.datamodule.data_train.close_neighbors

    def on_fit_start(self, trainer, pl_module):
        trainer.datamodule.data_train.close_neighbors = True

    def get_dists(self, samples):
        all_dists = []
        for sample in samples:
            # diff = samples-sample
            # dists = torch.norm(diff, dim=1)
            dists = 1-torch.cosine_similarity(sample, samples)
            all_dists.append(dists)
        return torch.stack(all_dists, dim=0)

    def extract_embeds(self, data_module, module):
        model = module.model
        model.eval()
        loader = data_module.val_dataloader()
        with torch.no_grad():
            if self.ensemble:
                embeds = [model(batch[0]) for batch in loader]
            else:
                if self.use_gpu:
                    model = model.cuda()
                    embeds = [model.get_embed(batch[0].cuda()) for batch in loader]
                else:
                    embeds = [model.get_embed(batch[0]) for batch in loader]
        embeds = torch.cat(embeds, dim=0).cpu()
        return embeds

if __name__=='__main__':
    from params import get_params
    params = get_params()
    data_module = ImlDataModule(params)
    data_module.ds[len(data_module.ds)-1]
    data_module.label_boot()
    data_module.next_batch()
    # data_module.transfer_samples([2,4])
    data_module.transfer_samples(list(np.arange(719)))
    data_module.next_batch()
    # data_module.transfer_samples([3,5])
    data_module.transfer_samples(list(np.arange(719)))
    print(data_module.get_current_session_name())
    data_module.ds.get_class_balance()
    print(len(data_module.ds), 'samples')
