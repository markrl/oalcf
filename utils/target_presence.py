import os
import sys
import glob
import numpy as np

def main(path):
    file_list = glob.glob(os.path.join(path,'*.npy'))
    labels = np.concatenate([np.load(ff) for ff in file_list])
    p_target = np.mean(labels)
    p_nontarget = 1 - p_target
    print(f'Target: {p_target*100:.2f}%')
    print(f'Nontarget: {p_nontarget*100:.2f}%')    

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    main(sys.argv[1])