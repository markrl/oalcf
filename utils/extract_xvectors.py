import os
import glob
import soundfile as sf
import resampy
import numpy as np
import torch
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier

from pdb import set_trace

def main():
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", 
                                        savedir="pretrained_models/spkrec-xvect-voxceleb",
                                        run_opts={"device":"cuda"})

    fs = 16000
    subdirs = ['train', 'dev', 'test']
    for subdir in subdirs:
        in_files = glob.glob(os.path.join(subdir, '*.mp3'))
        out_dir = os.path.join('xvectors', subdir)
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
                    output = model.encode_batch(torch.from_numpy(sig).float().unsqueeze(0).cuda())
                output = torch.mean(output[0], dim=0).cpu().numpy()
                np.save(out_file, output)
            except:
                continue
            
            
if __name__=='__main__':
    main()