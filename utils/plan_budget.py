import sys
import os
import pandas as pd
import numpy as np
import pickle

from src.dataset import VtdImlDataModule

from pdb import set_trace

def uniform(out_dir):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    avg_samples = 8
    inp_dir = 'output/vtd_best'
    subdirs = os.listdir(inp_dir)
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        n_sessions = len(sheet)
        n_samples = avg_samples*np.ones(n_sessions, dtype=int)
        os.mkdir(os.path.join(out_dir, subdir))
        out_file = os.path.join(out_dir, subdir, 'budget.txt')
        np.savetxt(out_file, n_samples)

def linear(out_dir, initial_value):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    avg_samples = 8
    inp_dir = 'output/vtd_best'
    subdirs = os.listdir(inp_dir)
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        n_sessions = len(sheet)
        total_samples = avg_samples*n_sessions
        sessions = np.arange(1,n_sessions+1)
        slope = n_sessions*(initial_value-avg_samples)/np.sum(sessions)
        n_samples = -slope*sessions + initial_value
        n_samples = np.round(n_samples)
        diff = total_samples-np.sum(n_samples)
        n_samples[1] += diff

        os.mkdir(os.path.join(out_dir, subdir))
        out_file = os.path.join(out_dir, subdir, 'budget.txt')
        np.savetxt(out_file, n_samples)
        print(f'Min {n_samples[-1]:.0f}')

def geometric_ratio(out_dir, lam_r):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    avg_samples = 8
    inp_dir = 'output/vtd_best'
    subdirs = os.listdir(inp_dir)
    first_se_samps = []
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        n_sessions = len(sheet)
        total_samples = n_sessions*avg_samples
        N = avg_samples*n_sessions
        r = 0.9
        a = 0.001
        lam_a = 0.5
        for ii in range(100):
            r_grad = 2*(a*r**N - r + 1 - a)*(N*a*r**(N-1) - 1)
            r = r - lam_r*r_grad
            a_grad = 2*(a*r**N - r + 1 - a)*(r**N - 1)
            a = a - lam_a*a_grad
        print(a*(1-r**N)/(1-r), r, a)
        n_samples = []
        for ii in range(n_sessions):
            new_value = total_samples*r**ii*a
            n_samples.append(new_value)
        n_samples = np.round(n_samples).astype(int)
        diff = total_samples-np.sum(n_samples)
        n_samples[0] += diff
        print(n_samples)

        os.mkdir(os.path.join(out_dir, subdir))
        out_file = os.path.join(out_dir, subdir, 'budget.txt')
        np.savetxt(out_file, n_samples)
        first_se_samps.append(n_samples[0])
    print(np.mean(first_se_samps))

def exponential_series(out_dir, initial_value=None):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    avg_samples = 8
    inp_dir = 'output/vtd_best'
    subdirs = os.listdir(inp_dir)
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        n_sessions = len(sheet)
        total_samples = avg_samples*n_sessions
        if initial_value is None:
            x = int(n_sessions/3)
        else:
            x = initial_value
        z = 0.1
        lam = 0.0000001
        for ii in range(100):
            z_grad = -2*(x*np.exp(-n_sessions*z) + (x+n_sessions*avg_samples)*np.exp(-z)-n_sessions*avg_samples)
            z_grad *= (n_sessions*x*np.exp(-n_sessions*z)+(x+n_sessions*avg_samples)*np.exp(-z))
            z = z-lam*z_grad
        
        n_samples = x*np.exp(-z*np.arange(1,n_sessions-1))
        n_samples = np.round(n_samples).astype(int)
        diff = total_samples-np.sum(n_samples)
        n_samples[0] += diff
        print(n_samples)

        os.mkdir(os.path.join(out_dir, subdir))
        out_file = os.path.join(out_dir, subdir, 'budget.txt')
        np.savetxt(out_file, n_samples)

def skewed_distribution(out_dir, a):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    avg_samples = 8
    inp_dir = 'output/vtd_best'
    subdirs = os.listdir(inp_dir)
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        n_sessions = len(sheet)
        total_samples = avg_samples*n_sessions
        n = np.arange(n_sessions)
        x = 1/np.sum(n**2/a**3*np.exp(-n**2/(2*a**2)))
        n_samples = total_samples*x*(n**2/a**3*np.exp(-n**2/(2*a**2)))
        n_samples = 0.5*n_samples + 0.5*avg_samples
        n_samples = np.round(n_samples).astype(int)
        diff = total_samples-np.sum(n_samples)
        n_samples[np.argmax(n_samples)] += diff
        print(n_samples)

        os.mkdir(os.path.join(out_dir, subdir))
        out_file = os.path.join(out_dir, subdir, 'budget.txt')
        np.savetxt(out_file, n_samples)

def oracle(out_dir):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    avg_samples = 8
    min_samples = 4
    params = pickle.load(open('/home/marklind/default_params.p', 'rb'))
    inp_dir = 'output/vtd_best'
    subdirs = os.listdir(inp_dir)
    for ii,subdir in enumerate(subdirs):
        params.env_name = subdir
        dm = VtdImlDataModule(params)
        dm.next_batch()
        weights = []
        while dm.current_batch < dm.n_batches:
            n_target = np.sum([dm.ds.get_label(ii) for ii in dm.data_test.active_idxs])
            weights.append(n_target)
            dm.next_batch()
        weights = np.array(weights)/np.sum(weights)
        total_samples = avg_samples*len(weights)
        n_samples = ((avg_samples-min_samples)/avg_samples)*total_samples*weights + min_samples
        n_samples = np.round(n_samples).astype(int)
        diff = total_samples-np.sum(n_samples)
        n_samples[np.argmax(n_samples)] += diff
        print(n_samples)
        print(np.sum(n_samples)-total_samples)

        os.mkdir(os.path.join(out_dir, subdir))
        out_file = os.path.join(out_dir, subdir, 'budget.txt')
        np.savetxt(out_file, n_samples)

if __name__=='__main__':
    if len(sys.argv) < 3:
        exit()
    elif sys.argv[1]=='uniform':
        uniform(sys.argv[2])
    elif sys.argv[1]=='linear':
        linear(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1]=='geometric':
        geometric_ratio(sys.argv[2], float(sys.argv[3]))
    elif sys.argv[1]=='exponential':
        if len(sys.argv)>3:
            exponential_series(sys.argv[2], int(sys.argv[3]))
        else:
            exponential_series(sys.argv[2])
    elif sys.argv[1]=='skew':
        skewed_distribution(sys.argv[2], float(sys.argv[3]))
    elif sys.argv[1]=='oracle':
        oracle(sys.argv[2])