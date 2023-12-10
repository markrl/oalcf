import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def main(budget_paths, label_names=None):
    budget_paths = budget_paths.strip().split(',')
    if label_names is not None:
        label_names = label_names.strip().split(',')
    else:
        label_names = [nn+1 for nn in range(len(budget_paths))]
    for ii,ff in enumerate(budget_paths):
        if 'budget.txt' not in ff:
            ff = os.path.join(ff, 'budget.txt')
        trace = np.genfromtxt(ff)
        trace = np.cumsum(trace)
        plt.plot(trace, label=label_names[ii])
    plt.xlabel('Session')
    plt.ylabel('Accumulated samples')
    plt.legend()
    plt.savefig('/home/marklind/research/AL/vtd_iml/plotting/plot_out/budget.png')
    plt.clf()

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    elif len(sys.argv)==2:
        main(sys.argv[1])
    else:
        main(sys.argv[1], sys.argv[2])