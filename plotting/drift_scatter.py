import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(csv_path, metric=None):
    if metric is None:
        metric = 'dcf'
    sheet = pd.read_csv(csv_path)
    corr = np.corrcoef(sheet[metric], sheet['drift'])[0,1]
    plt.scatter(sheet[metric], sheet['drift'], alpha=0.5)
    plt.title(f'PCC: {corr:.4f}')
    plt.xlabel(metric.upper())
    plt.ylabel('Drift')
    plt.savefig('/home/marklind/research/AL/vtd_iml/plotting/plot_out/drift_scatter.png')
    plt.clf()
    print(corr)

if __name__=='__main__':
    if len(sys.argv)<2:
        exit()
    if len(sys.argv)==2:
        main(sys.argv[1])
    else:
        main(sys.argv[1], sys.argv[2])