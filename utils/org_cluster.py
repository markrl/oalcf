import os
import glob
import random
import numpy as np

from pdb import set_trace

def main():
    out_dir = 'orders'
    for subdir in ['train', 'dev', 'test']:
        print(subdir)
        file_list = glob.glob(os.path.join('wavlm', subdir, '*.npy'))
        class_dict = {}
        class_list = []
        for ii,ff in enumerate(file_list):
            file_class = ff.split('_')[2]
            if file_class in class_dict:
                class_dict[file_class].append(ii)
            else:
                class_dict[file_class] = [ii]
            class_list.append(file_class)

        # dirichlet distribution
        cluster_idxs = []
        dirichlet_numchunks = len(class_dict.keys())
        delta = 1

        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
        min_size = -1
        N = len(class_list)
        min_size_thresh = 10
        while min_size < min_size_thresh:  # prevent any chunk having too little data
            idx_batch = [[] for _ in range(dirichlet_numchunks)]
            idx_batch_cls = [[] for _ in range(dirichlet_numchunks)] # contains data per each class
            for k in class_dict.keys():
                cl_labels = np.array(class_list)
                idx_k = np.where(cl_labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(delta, dirichlet_numchunks))

                # balance
                proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in
                                        zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

                # store class-wise data
                for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                    idx_j.append(idx)

        # create temporally correlated toy dataset by shuffling classes
        for chunk in idx_batch_cls:
            cls_seq = list(range(dirichlet_numchunks))
            np.random.shuffle(cls_seq)
            for cls in cls_seq:
                idx = chunk[cls]
                cluster_idxs.extend([i for i in idx])

        # trim data if num_sample is smaller than the original data size
        cluster_idxs = cluster_idxs[:len(file_list)]
               
        out_file = open(os.path.join(out_dir, f'cluster_order_{subdir}'), 'w')
        for ii in cluster_idxs:
            out_file.write(f'{os.path.basename(file_list[ii])[:-4]}\n')

if __name__=='__main__':
    main()