import os
import glob
import random
from tqdm import tqdm

def main():
    out_dir = 'orders'
    for subdir in ['train', 'dev', 'test']:
        file_list = glob.glob(os.path.join('wavlm', subdir, '*.npy'))
        random.shuffle(file_list)
        out_file = open(os.path.join(out_dir, f'random_order_{subdir}'), 'w')
        print(subdir)
        for ff in tqdm(file_list):
            out_file.write(f'{os.path.basename(ff)[:-4]}\n')

if __name__=='__main__':
    main()