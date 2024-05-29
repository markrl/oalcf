import os
import glob
import soundfile as sf
import resampy
import numpy as np
import torch
from transformers import WavLMModel
from tqdm import tqdm

from pdb import set_trace

def main():
    # WavLM setup
    model = WavLMModel.from_pretrained('microsoft/wavlm-large')
    model = model.cuda()
    configuration = model.config

    fs = 16000
    subdirs = ['train', 'dev', 'test']
    for subdir in subdirs:
        in_files = glob.glob(os.path.join(subdir, '*.mp3'))
        out_dir = os.path.join('wavlm', subdir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        print(subdir)
        for ff in tqdm(in_files):
            out_file = os.path.join(out_dir, os.path.basename(ff).replace('.mp3', '.npy'))
            if os.path.exists(out_file):
                continue
            try:
                sig,orig_fs = sf.read(ff)
                if orig_fs != fs:
                    sig = resampy.resample(sig, orig_fs, fs)
                with torch.no_grad():
                    output = model(torch.from_numpy(sig).float().unsqueeze(0).cuda())
                output = torch.mean(output.last_hidden_state[0], dim=0).cpu().numpy()
                np.save(out_file, output)
            except:
                continue
            
            
if __name__=='__main__':
    main()