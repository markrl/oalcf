import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(csv_paths, metric=None, label_names=None, outpath=None):
    if metric is None:
        metric = 'dcf'
    if outpath is None:
        outpath = '/home/marklind/research/AL/vtd_iml/plotting/plot_out/comp_env.png'
    csv_files = csv_paths.split(',')
    min_val, max_val = np.inf, 0
    for ii,ff in enumerate(csv_files):
        if os.path.basename(ff) != 'scores.csv':
            ff = os.path.join(ff, 'scores.csv')
        sheet = pd.read_csv(ff)
        if metric=='dcf':
            scores = np.array(sheet['cum_dcf'])
        elif metric=='dcf_ind':
            scores = np.array(sheet['dcf'])
        elif metric=='fnr':
            fns = np.cumsum(sheet['fns'])
            ps = np.cumsum(sheet['ps'])
            scores = fns/ps
        elif metric=='fpr':
            fns = np.cumsum(sheet['fps'])
            ps = np.cumsum(sheet['ns'])
            scores = fps/ns
        elif metric=='new_samples':
            scores = np.array(sheet['n_samples'])
            scores = np.diff(scores, prepend=8)
        else:
            scores = np.array(sheet[metric])
        scores[np.isnan(scores)] = 0
        print(f'{scores[-1]:.4f}')
        if label_names is None:
            plt.plot(scores, label=ff.split('/')[-2])
        else:
            plt.plot(scores, label=label_names.split(',')[ii])
        if np.min(scores) < min_val:
            min_val = np.min(scores)
        if np.max(scores) > max_val:
            max_val = np.max(scores)
    # Apartment
    drift_sessions = [15, 18, 25, 40, 103, 122, 125, 128]
    # Hotel
    # drift_sessions = [8, 33, 48, 52, 73, 101, 121, 126]
    # Office
    # drift_sessions = [41, 62, 113, 127]
    # Rm1
    # drift_sessions = [3, 14, 47]
    # Rm2
    # drift_sessions = [7, 14, 60, 83, 90, 93]
    # Rm3
    # drift_sessions = [8, 15, 26, 71]
    # Rm4
    # drift_sessions = [5, 26, 35, 66, 74]
    for ii,ss in enumerate(drift_sessions):
        if ii==0:
            plt.plot([ss, ss], [min_val, max_val], 'k--', label='drift')
        else:
            plt.plot([ss, ss], [min_val, max_val], 'k--')
    plt.xlabel('Session')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.savefig(outpath)
    plt.clf()


if __name__=='__main__':
    if len(sys.argv)<2:
        exit()
    if len(sys.argv)==2:
        main(sys.argv[1])
    elif len(sys.argv)==3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])