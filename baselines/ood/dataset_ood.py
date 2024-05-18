import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import numpy as np
from statistics import mean
from copy import deepcopy

from pdb import set_trace

class SupervisedDataModule(LightningDataModule):
    def __init__(self, params):
        self.params = params
        super().__init__()
        # Form dataset
        if 'VTD' in params.feat_root:
            self.data_train = VtdData(params, 'train')
            self.data_val = VtdData(params, 'val')
            self.data_test = VtdData(params, 'test')
        elif 'LID' in params.feat_root:
            self.data_train = LidData(params, 'train')
            self.data_val = LidData(params, 'dev')
            self.data_test = LidData(params, 'test')

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
            dataset=self.data_val,
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

    def get_class_counts(self):
        labels = torch.FloatTensor([self.data_train.get_label(ii) for ii in range(len(self.data_train))])
        n_target = torch.sum(labels)
        n_nontarget = len(labels)-n_target
        return n_target,n_nontarget
    
    def get_class_balance(self):
        n_target,n_nontarget = self.get_class_counts()
        p_target = n_target/(n_target+n_nontarget)
        p_nontarget = n_nontarget/(n_target+n_nontarget)
        return p_target,p_nontarget

    def __len__(self):
        return len(self.data_train)

class VtdData(Dataset):
    def __init__(self, 
                 params,
                 fold,
                 ):
        super().__init__()
        self.params = params
        self.fold = fold
        self.feat_roots = params.feat_root.split(',')
        self.feat_names = [os.listdir(rr)[0].split('_')[3] for rr in self.feat_roots]
        self.feat_files = glob.glob(os.path.join(self.feat_roots[0], '*.npy'))
        for ii in range(len(self.feat_files)):
            rm,se,mc,_,splt = os.path.basename(self.feat_files[ii]).split('_')
            self.feat_files[ii] = '_'.join((rm, se, mc, '{}', splt))
        if params.corpus == 'sri':
            if fold == 'train':
                self.feat_files = [ff for ff in self.feat_files if 'rm' in ff and 'rm3' not in ff]
            elif fold == 'val':
                self.feat_files = [ff for ff in self.feat_files if 'rm3' in ff]
            elif fold == 'test':
                self.feat_files = [ff for ff in self.feat_files if 'rm' not in ff]
        elif params.corpus == 'lb':
            if fold == 'train':
                self.feat_files = [ff for ff in self.feat_files if 'rm' not in ff and 'office' not in ff]
            elif fold == 'val':
                self.feat_files = [ff for ff in self.feat_files if 'office' in ff]
            elif fold == 'test':
                self.feat_files = [ff for ff in self.feat_files if 'rm' in ff]
        self.feat_files.sort()

        self.label_files = []
        bad_files = []
        for ff in self.feat_files:
            rm,se,_,_,splt = os.path.basename(ff).split('_')
            label_file = os.path.join(params.ann_root, '_'.join((rm,se,splt)))
            self.label_files.append(label_file)
            if not np.all([os.path.exists(os.path.join(rr, ff).format(nn)) for rr,nn in zip(self.feat_roots, self.feat_names)]+[os.path.exists(label_file)]):
                bad_files.append((label_file, ff))
        
        for lf,ff in bad_files:
            self.label_files.remove(lf)
            self.feat_files.remove(ff)

    def __len__(self):
        return len(self.label_files)*self.params.samples_per_batch

    def __getitem__(self, index):
        label1 = self.get_label(index)
        feat1 = self.get_sample_with_context(index)
        if self.fold!='train':
            return torch.from_numpy(feat1).float(), label1, 0, 0
        else:
            idx2 = np.random.randint(low=0, high=len(self))
            label2 = self.get_label(idx2)
            feat2 = self.get_sample_with_context(idx2)
            return (torch.from_numpy(feat1).float(), label1, 
                    torch.from_numpy(feat2).float(), label2)

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


class LidData(Dataset):
    def __init__(self, 
                 params,
                 fold,
                 ):
        super().__init__()
        self.params = params
        self.fold = fold
        self.feat_roots = params.feat_root.split(',')
        self.feat_names = [rr.split('/')[4] for rr in self.feat_roots]
        self.feat_files = glob.glob(os.path.join(self.feat_roots[0], fold, '*.npy'))
        self.feat_files = [os.path.basename(ff)[:-4] for ff in self.feat_files]
        self.labels = [1*(ff.split('_')[2]==params.lid_target) for ff in self.feat_files]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label1 = self.get_label(index)
        feat1 = np.concatenate([np.load(os.path.join(rr, self.fold, f'{self.feat_files[index]}.npy')) for rr in self.feat_roots])
        if self.fold!='train':
            return torch.from_numpy(feat1).float(), label1, 0, 0
        else:
            idx2 = np.random.randint(low=0, high=len(self))
            label2 = self.get_label(idx2)
            feat2 = np.concatenate([np.load(os.path.join(rr, self.fold, f'{self.feat_files[idx2]}.npy')) for rr in self.feat_roots])
            return (torch.from_numpy(feat1).float(), label1, 
                    torch.from_numpy(feat2).float(), label2)            

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


class SupervisedEvalDataModule(LightningDataModule):
    def __init__(self, params):
        self.params = params
        super().__init__()
        # Form dataset
        if 'VTD' in params.feat_root:
            self.data_test = VtdEvalData(params)
        elif 'LID' in params.feat_root:
            self.data_test = LidData(params)

    def setup(self, stage):
        return

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=False,
        )

    def get_class_counts(self):
        labels = torch.FloatTensor([self.data_test.get_label(ii) for ii in range(len(self.data_test))])
        n_target = torch.sum(labels)
        n_nontarget = len(labels)-n_target
        return n_target,n_nontarget
    
    def get_class_balance(self):
        n_target,n_nontarget = self.get_class_counts()
        p_target = n_target/(n_target+n_nontarget)
        p_nontarget = n_nontarget/(n_target+n_nontarget)
        return p_target,p_nontarget

    def __len__(self):
        return len(self.data_test)


class VtdEvalData(Dataset):
    def __init__(self, 
                 params,
                 ):
        super().__init__()
        self.params = params
        command_path = os.path.join(params.eval_run, 'command.txt')
        with open(command_path, 'r') as f:
            command = f.readline().split(' ')
        env_name = command[np.where([cc=='--env_name' for cc in command])[0][0]+1]
        rm, mc = env_name.split('_')

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

        self.active_idxs = [nn for nn in range(len(self.feat_files)*self.params.samples_per_batch)]
        al_path = os.path.join(params.eval_run, 'al_samples.txt')
        with open(al_path, 'r') as f:
            used_idxs = f.readlines()
        for uu in used_idxs:
            af,idx = uu.strip().split(',')
            idx = int(idx)
            rm,se,mc,splt = af.split('_')
            af = '_'.join((rm,se,mc,'{}',splt)) + '.npy'
            al_idx = np.where([ff==af for ff in self.feat_files])[0][0]*self.params.samples_per_batch + idx
            self.active_idxs.remove(al_idx)

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, index):
        index = self.active_idxs[index]
        label1 = self.get_label(index)
        feat1 = self.get_sample_with_context(index)
        return torch.from_numpy(feat1).float(), label1, 0, 0

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


if __name__=='__main__':
    from params_ood import get_params
    params = get_params()
    if params.eval_run is None:
        data_module = VtdDataModule(params)
        print('Train length:',len(data_module.data_train.label_files))
        print('Val length:',len(data_module.data_val.label_files))
        print('Test length',len(data_module.data_test.label_files))
        print(data_module.data_train[0])
    else:
        data_module = VtdEvalDataModule(params)
        print(f'Original length: {len(data_module.data_test.feat_files):d} hours')
        print(f'Reduced length: {len(data_module)*5/3600:.1f} hours')
