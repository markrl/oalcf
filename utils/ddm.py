import os
import sys
import numpy as np
import torch
import torch.nn.functional as f
import scipy.stats as stats
from copy import deepcopy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from pdb import set_trace


class LDD_DIS:
    def __init__(self, 
                neighborhood_ratio=0.1, 
                significance_level=0.05, 
                dist_fn='l2', 
                zero_mean=True, 
                log_compress=False):
        '''
        Based on the paper "Regional Concept Drift Detection and Density Synchronized Drift Adaptation" by Anjin Liu, Yiliao Song, Guangquan Zhang, and Jie Lu
        https://opus.lib.uts.edu.au/bitstream/10453/126374/1/0317.pdf

        See Algorithm 1 in the paper for the steps
        '''
        self.rho = neighborhood_ratio
        self.alpha = significance_level
        self.zero_mean = zero_mean
        self.log_compress = log_compress
        if dist_fn=='l2':
            self.dist_fn = lambda d_i,D:torch.sqrt(torch.sum((D-d_i)**2, dim=1))
        elif dist_fn=='cos':
            self.dist_fn = lambda d_i,D:1-f.cosine_similarity(d_i,D)
        else:
            print('Invalid `dist_fn`')
            exit()

    def analyze_drift(self, D1, D2):
        # 1
        D = torch.cat([D1, D2], dim=0)
        D1_idxs = set(torch.arange(len(D1)).tolist())
        D2_idxs = set((torch.arange(len(D2))+len(D1_idxs)).tolist())
        k = int(self.rho*(len(D)))

        # 2-4
        D_knn = []
        for ii in range(len(D)):
            D_knn.append(self.findKNN(ii, D, k))
        
        # 5
        shuffled_idxs = torch.randperm(len(D)).tolist()
        D1_prime_idxs = set(shuffled_idxs[:int(len(D)/2)])
        D2_prime_idxs = set(shuffled_idxs[int(len(D)/2):])

        # 6-11
        delta_primes = []
        for ii in D1_prime_idxs:
            num = len(D_knn[ii].intersection(D2_prime_idxs))
            denom = len(D_knn[ii].intersection(D1_prime_idxs))
            denom = denom+1 if denom==0 else denom
            if self.log_compress:
                delta_prime_i = torch.log(torch.tensor(num/denom))
            else:
                delta_prime_i = num/denom-1
            delta_primes.append(delta_prime_i)
        for ii in D2_prime_idxs:
            num = len(D_knn[ii].intersection(D1_prime_idxs))
            denom = len(D_knn[ii].intersection(D2_prime_idxs))
            denom = denom+1 if denom==0 else denom
            if self.log_compress:
                delta_prime_i = torch.log(torch.tensor(num/denom))
            else:
                delta_prime_i = num/denom-1
            delta_primes.append(delta_prime_i)
        
        # 12
        delta_prime_std = torch.std(torch.FloatTensor(delta_primes))
        delta_prime_mean = 0 if self.zero_mean else torch.mean(torch.FloatTensor(delta_primes))
        theta_dec = stats.norm.ppf(self.alpha, delta_prime_mean, delta_prime_std)
        # 13
        theta_inc = stats.norm.ppf(1-self.alpha, delta_prime_mean, delta_prime_std)

        # 14-31
        D1_dec, D1_sta, D1_inc, D2_dec, D2_sta, D2_inc = [], [], [], [], [], []
        deltas = []
        for ii in D1_idxs:
            num = len(D_knn[ii].intersection(D2_idxs))
            denom = len(D_knn[ii].intersection(D1_idxs))
            denom = denom+1 if denom==0 else denom
            if self.log_compress:
                delta_i = torch.log(torch.tensor(num/denom))
            else:
                delta_i = num/denom-1
            deltas.append(delta_i)
            if delta_i < theta_dec:
                D1_dec.append(ii)
            elif delta_i >= theta_inc:
                D1_inc.append(ii)
            else:
                D1_sta.append(ii)
        for ii in D2_idxs:
            num = len(D_knn[ii].intersection(D1_idxs))
            denom = len(D_knn[ii].intersection(D2_idxs))
            denom = denom+1 if denom==0 else denom
            if self.log_compress:
                delta_i = torch.log(torch.tensor(num/denom))
            else:
                delta_i = num/denom-1
            deltas.append(delta_i)
            if delta_i < theta_dec:
                D2_dec.append(ii)
            elif delta_i >= theta_inc:
                D2_inc.append(ii)
            else:
                D2_sta.append(ii)

        # 32
        D_drift = [D1_dec, D1_sta, D1_inc, D2_dec, D2_sta, D2_inc]
        return D_drift, deltas
    
    def findKNN(self, ii, D, k):
        d_i = D[ii]
        dists = self.dist_fn(d_i, D)
        sorted_idxs = torch.argsort(dists)
        sorted_idxs = sorted_idxs[sorted_idxs!=ii]
        D_knn_i = sorted_idxs[:k]
        return set(D_knn_i.tolist())


def extract_env(env_path, se_len):
    import pandas as pd
    clips_per_se = int(se_len*6)
    sheet = pd.read_csv(env_path, names=['files'])
    root = os.path.join('/data1', 'sonyc_clap')
    
    ses = []
    start = 0
    while start+clips_per_se < len(sheet):
        current_files = sheet['files'][start:start+clips_per_se]
        current_se = [torch.load(os.path.join(root, ff.replace('.wav', '.pt'))) for ff in current_files]
        ses.append(torch.stack(current_se))
        start += clips_per_se
    return ses


def plot_scatter(train_data, test_data, D_drift, plot_test=False):
    train_data -= torch.mean(train_data, dim=0)
    test_data -= torch.mean(test_data, dim=0)
    train_data = train_data.float()
    test_data = test_data.float()
    if train_data.shape[1] > 2:
        D = torch.cat([train_data, test_data], dim=0)
        V = torch.linalg.svd(D, False)[2][:2].T
        train_data = torch.mm(train_data,V)
        test_data = torch.mm(test_data,V)
    plt.scatter(train_data[:,0], train_data[:,1], s=10, marker='o', label='train')
    plt.scatter(test_data[:,0], test_data[:,1], s=15, marker='+', label='test')
    plt.legend()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('dist_plot')
    plt.clf()

    plt.scatter(train_data[:,0], train_data[:,1], c='#FFFFFF')
    plt.scatter(test_data[:,0], test_data[:,1], c='#FFFFFF')
    if not plot_test:
        decs = D_drift[0]
        incs = D_drift[2]
        plt.scatter(train_data[decs,0], train_data[decs,1], s=10, marker='o', label='dec')
        plt.scatter(train_data[incs,0], train_data[incs,1], s=15, marker='+', label='inc')
    else:
        decs = D_drift[3]
        incs = D_drift[5]
        plt.scatter(test_data[decs,0], test_data[decs,1], s=10, marker='o', label='dec')
        plt.scatter(test_data[incs,0], test_data[incs,1], s=15, marker='+', label='inc')
    plt.legend()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('drift_plot')
    plt.clf()


def plot_env(drift_counts):
    import numpy as np
    names = ['D1_dec', 'D1_sta', 'D1_inc', 'D2_dec', 'D2_sta', 'D2_inc']
    x = np.arange(len(names))
    width = 0.7/len(drift_counts)
    multiplier = 0

    fig, ax = plt.subplots()

    for ses, count in drift_counts.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, count, width, label=ses)
        multiplier += 1

    ax.set_xticks(x + width*(len(drift_counts)-1)/2, names)
    ax.set_ylabel('Count')
    ax.set_title('Density Drifts')
    
    plt.savefig('bar')
    plt.clf()


class NnDdm:
    def __init__(self, dist_fn=None, reduction='max'):
        self.reduction = reduction
        if dist_fn is None: # Euclidean
            self.dist_fn = lambda x,y:np.sqrt(np.sum((x-y)**2))
        else:
            self.dist_fn = dist_fn

    def get_dist(self, known_data, new_data):
        known_data_points = []
        for dd in known_data:
            known_data_points.append(dd[0])
        known_data_points = torch.stack(known_data_points, dim=0).numpy() 

        new_data_points = []
        for dd in new_data:
            new_data_points.append(dd[0])
        new_data_points = torch.stack(new_data_points, dim=0).numpy()

        nn_dists = []
        for nd in new_data:
            min_dist = np.inf
            for kd in known_data:
                dist = self.dist_fn(kd, nd)
                if dist < min_dist:
                    min_dist = dist
            nn_dists.append(min_dist)
        nn_dists = np.array(nn_dists)
        if self.reduction=='max':
            return np.max(nn_dists)
        elif self.reduction=='mean':
            return np.mean(nn_dists)
        else:
            return NotImplemented
    

class NcDdm:
    def __init__(self, n_clusters=10, dist_fn=None, reduction='max'):
        self.reduction = reduction
        self.n_clusters = n_clusters
        if dist_fn is None: # Euclidean
            self.dist_fn = lambda x,y:np.sqrt(np.sum((x-y)**2))
            self.dist_name = 'euclidean'
        elif dist_fn=='std':
            def dist_(x,y,stds_x):
                if np.linalg.norm(x-y)==0:
                    return 0
                proj_vec = x-(x-y)/np.linalg.norm(x-y)*stds_x
                contour_dist = np.sqrt(np.sum((x-proj_vec)**2))
                vec_dist = np.sqrt(np.sum((x-y)**2))
                return np.maximum(vec_dist-contour_dist, 0)

            self.dist_fn = dist_
            self.dist_name = dist_fn
        else:
            self.dist_fn = dist_fn
            self.dist_name = None

    def get_dist(self, known_data, new_data):
        known_data_points = []
        for dd in known_data:
            known_data_points.append(dd[0])
        known_data_points = torch.stack(known_data_points, dim=0).numpy() 

        new_data_points = []
        for dd in new_data:
            new_data_points.append(dd[0])
        new_data_points = torch.stack(new_data_points, dim=0).numpy()

        known_model = KMeans(self.n_clusters, random_state=0, n_init='auto').fit(known_data_points)
        new_model = KMeans(self.n_clusters, random_state=0, n_init='auto').fit(new_data_points)
        
        known_means = []
        known_stds = []
        for ii in range(self.n_clusters):
            cluster_samples = known_data_points[known_model.labels_==ii]
            known_means.append(np.mean(cluster_samples, axis=0))
            known_stds.append(np.std(cluster_samples, axis=0))

        new_means = []
        for ii in range(self.n_clusters):
            cluster_samples = new_data_points[new_model.labels_==ii]
            new_means.append(np.mean(cluster_samples, axis=0))

        nc_dists = []
        cluster_sizes = []
        for ii,nc in enumerate(new_means):
            min_dist = np.inf
            for jj,kc in enumerate(known_means):
                if self.dist_name=='std':
                    dist = self.dist_fn(kc, nc, known_stds[jj])
                else:
                    dist = self.dist_fn(kc, nc)
                if dist < min_dist:
                    min_dist = dist
            nc_dists.append(min_dist)
            cluster_sizes.append(np.sum(1*(new_model.labels_==ii)))
        nc_dists = np.array(nc_dists)
        cluster_sizes = np.array(cluster_sizes)
        if self.reduction=='max':
            return np.max(nc_dists)
        elif self.reduction=='weighted':
            return np.sum((nc_dists*cluster_sizes)/np.sum(cluster_sizes))
        elif self.reduction=='mean':
            return np.mean(nc_dists)
        else:
            return NotImplemented


if __name__=='__main__':
    if len(sys.argv) != 4:
        print('Please run this command: `python ddm.py [path/to/env] [session_length_minutes] [neighborhood_ratio]')
        exit()
    batches = extract_env(sys.argv[1], int(sys.argv[2]))
    ddm = LDD_DIS(neighborhood_ratio=float(sys.argv[3]), dist_fn='l2')
    drift_counts = {}
    for ii in range(len(batches)-1):
        D_drift = ddm.analyze_drift(batches[ii], batches[ii+1])[0]
        drift_counts[f'{ii+1}/{ii+2}'] = [len(dd) for dd in D_drift]
        print(f'Session {ii+1:d}:\tSession {ii+2:d}:')
        print(f'{len(D_drift[0]):d} decreased\t{len(D_drift[3]):d} decreased')
        print(f'{len(D_drift[1]):d} stable\t{len(D_drift[4]):d} stable')
        print(f'{len(D_drift[2]):d} increased\t{len(D_drift[5]):d} increased')
        print()
    plot_env(drift_counts)
    plot_scatter(batches[-2], batches[-1], D_drift)