import sys
import os
import pandas as pd
import numpy as np
import pickle

from src.dataset import ImlDataModule

from pdb import set_trace

def uniform(inp_dir, out_dir):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    subdirs = os.listdir(inp_dir)
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        avg_samples = np.mean(sheet['n_al'])
        n_sessions = len(sheet)
        total_samples = np.sum(sheet['n_al'])
        n_samples = avg_samples*np.ones(n_sessions, dtype=int)
        os.mkdir(os.path.join(out_dir, subdir))
        out_file = os.path.join(out_dir, subdir, 'budget.txt')
        np.savetxt(out_file, n_samples)

def frontheavy(inp_dir, out_dir):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    subdirs = os.listdir(inp_dir)
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        avg_samples = np.mean(sheet['n_al'])
        n_sessions = len(sheet)
        total_samples = np.sum(sheet['n_al'])
        n_samples = np.zeros(n_sessions)
        ii = 0
        while total_samples > 0:
            n_samples[ii] = np.minimum(720, total_samples)
            total_samples -= n_samples[ii]
            ii += 1

        os.mkdir(os.path.join(out_dir, subdir.split('_')[0]))
        out_file = os.path.join(out_dir, subdir.split('_')[0], 'budget.txt')
        np.savetxt(out_file, n_samples)

def linear(inp_dir, out_dir, initial_value):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    subdirs = os.listdir(inp_dir)
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        avg_samples = np.mean(sheet['n_al'])
        n_sessions = len(sheet)
        total_samples = np.sum(sheet['n_al'])
        sessions = np.arange(1,n_sessions+1)
        slope = n_sessions*(initial_value-avg_samples)/np.sum(sessions)
        n_samples = -slope*sessions + initial_value
        n_samples = np.round(n_samples)
        diff = total_samples-np.sum(n_samples)
        n_samples[1] += diff

        os.mkdir(os.path.join(out_dir, subdir.split('_')[0]))
        out_file = os.path.join(out_dir, subdir.split('_')[0], 'budget.txt')
        np.savetxt(out_file, n_samples)
        print(f'Min {n_samples[-1]:.0f}')

def geometric_ratio(inp_dir, out_dir, lam_r):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    # inp_dir = 'output/vtd_best'
    subdirs = os.listdir(inp_dir)
    first_se_samps = []
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        avg_samples = np.mean(sheet['n_al'])
        n_sessions = len(sheet)
        total_samples = np.sum(sheet['n_al'])
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

def exponential_series(inp_dir, out_dir, initial_value=None):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    # inp_dir = 'output/vtd_best'
    subdirs = os.listdir(inp_dir)
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        avg_samples = np.mean(sheet['n_al'])
        n_sessions = len(sheet)
        total_samples = np.sum(sheet['n_al'])
        if initial_value is None:
            # x = int(total_samples/3)
            x = int(0.03*total_samples)
            p = 0.03
        else:
            x = initial_value
        # z = 0.1
        # lam = 0.0000001
        # for ii in range(100):
        #     # z_grad = -2*(x*np.exp(-n_sessions*z) + (x+n_sessions*avg_samples)*np.exp(-z)-n_sessions*avg_samples)
        #     # z_grad *= (n_sessions*x*np.exp(-n_sessions*z)+(x+n_sessions*avg_samples)*np.exp(-z))
        #     z_grad = 2*(x*np.exp(-n_sessions*z) - total_samples*np.exp(-z))
        #     z_grad *= x/n_sessions*np.exp(-n_sessions*z) - total_samples*np.exp(-z)
        #     z = z-lam*z_grad
        
        # n_samples = x*np.exp(-z*np.arange(n_sessions))
        k = np.arange(n_sessions)
        n_samples = (1-p)**k*p*total_samples
        n_samples = np.round(n_samples).astype(int)
        diff = total_samples-np.sum(n_samples)
        n_samples[0] += diff
        print(n_samples)

        os.mkdir(os.path.join(out_dir, subdir.split('_')[0]))
        out_file = os.path.join(out_dir, subdir.split('_')[0], 'budget.txt')
        np.savetxt(out_file, n_samples)

def skewed_distribution(inp_dir, out_dir, a):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    subdirs = os.listdir(inp_dir)
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        avg_samples = np.mean(sheet['n_al'])
        n_sessions = len(sheet)
        total_samples = np.sum(sheet['n_al'])
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

def specific_sessions(inp_dir, out_dir, se_list):
    if '.' in se_list:
        se_list = se_list.split(',')
        se_list = [float(ss) for ss in se_list]
    else:
        se_list = se_list.split(',')
        se_list = [int(ss) for ss in se_list]

    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    subdirs = os.listdir(inp_dir)
    for ii,subdir in enumerate(subdirs):
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        avg_samples = np.mean(sheet['n_al'])
        n_sessions = len(sheet)
        total_samples = np.sum(sheet['n_al'])
        per_session = int(total_samples/len(se_list))
        per_session = np.minimum(per_session, 711)
        n_samples = np.zeros(n_sessions)
        if type(se_list[0])==float:
            ses = np.array([int(n_sessions*ss) for ss in se_list])
        else:
            ses = np.array(se_list)
        n_samples[ses] = per_session
        diff = total_samples-np.sum(n_samples)
        n_samples[ses[0]+1] += diff
        print(n_samples)

        os.mkdir(os.path.join(out_dir, subdir))
        out_file = os.path.join(out_dir, subdir, 'budget.txt')
        np.savetxt(out_file, n_samples)        

def oracle(inp_dir, out_dir):
    if 'budgets/' not in out_dir:
        out_dir = os.path.join('budgets', out_dir)
    os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    avg_samples = 8
    min_samples = 4
    params = pickle.load(open('/home/marklind/default_params.p', 'rb'))
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
    if len(sys.argv) < 4:
        exit()
    elif sys.argv[1]=='uniform':
        uniform(sys.argv[2], sys.argv[3])
    elif sys.argv[1]=='frontheavy':
        frontheavy(sys.argv[2], sys.argv[3])
    elif sys.argv[1]=='linear':
        linear(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1]=='geometric':
        geometric_ratio(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    elif sys.argv[1]=='exponential':
        if len(sys.argv)>4:
            exponential_series(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        else:
            exponential_series(sys.argv[2], sys.argv[3])
    elif sys.argv[1]=='skew':
        skewed_distribution(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    elif sys.argv[1]=='oracle':
        oracle(sys.argv[2], sys.argv[3])
    elif sys.argv[1]=='sessions':
        specific_sessions(sys.argv[2], sys.argv[3], sys.argv[4])