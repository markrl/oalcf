import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(in_dir):
    score_path = os.path.join(in_dir, 'scores.csv')
    sheet = pd.read_csv(score_path)
    if 'drift' not in sheet.keys():
        exit()
    contents = np.array(sheet['drift'])
    plt.plot(contents, 'o')
    plt.xlabel('Session')
    plt.ylabel('Drift presence')
    plt.savefig('plotting/plot_out/drift_plot.png')
    print(np.mean(contents))


if __name__=='__main__':
    if len(sys.argv)<2:
        exit()
    main(sys.argv[1])