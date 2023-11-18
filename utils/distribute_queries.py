import sys
import os
import pandas as pd
import numpy as np

def main(inp_dir):
    avg_samples = 8
    min_samples = 4
    subdirs = os.listdir(inp_dir)
    for subdir in subdirs:
        sheet = pd.read_csv(os.path.join(inp_dir, subdir, 'scores.csv'))
        budget = avg_samples*len(sheet)
        fns = np.array(sheet['fns'])
        fps = np.array(sheet['fps'])
        weights = 0.25*fps + 0.75*fns
        weights /= np.sum(weights)
        mult = (avg_samples-min_samples)*len(sheet)
        n_samples = np.round(weights*mult+min_samples).astype(int)
        diff = budget - np.sum(n_samples)
        n_samples[np.argmax(n_samples)] += diff

        out_file = os.path.join(inp_dir, subdir, 'budget.txt')
        np.savetxt(out_file, n_samples)

if __name__=='__main__':
    if len(sys.argv)<=1:
        exit()
    main(sys.argv[1])