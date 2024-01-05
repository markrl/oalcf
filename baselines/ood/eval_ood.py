import torch
import numpy as np
import os
import sys

from pytorch_lightning import Trainer

from src.module import VtdModule
from baselines.ood.dataset_ood import VtdEvalDataModule
from baselines.ood.params_ood import get_params
from utils.utils import write_header, write_session

from pdb import set_trace

def main():
    # Set pytorch precision
    torch.set_float32_matmul_precision('medium')
    # Get and handle parameters
    params = get_params()
    # Set up input and output directories
    model_path = os.path.join('baselines/ood/output', params.ckpt_name, 'state_dict.pt')
    out_dir = params.eval_run
    out_file = os.path.join(out_dir, 'baseline_ood_scores.csv')
    write_header(out_file, [], False)
    # Initialize lightning data module and lightning module
    data_module = VtdEvalDataModule(params)
    module = VtdModule(params)
    module.model.load_state_dict(torch.load(model_path))
    # Instantiate trainer
    trainer = Trainer(
        accelerator='gpu' if params.gpus>0 else 'cpu',
        devices=[0],
        overfit_batches=params.overfit_batches,
        max_epochs=params.max_epochs,
        check_val_every_n_epoch=params.val_every_n_epochs,
        logger=None,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )
    # Evaluate on test corpus
    test_results = trainer.test(module, data_module)

    # Write results to file
    fps, fns, ps, ns = [], [], [], []
    train_info = np.genfromtxt(os.path.join('baselines/ood/output', params.ckpt_name, 'scores.csv'), delimiter=',', skip_header=1)
    train_len = int(train_info[10])
    if not params.debug:
        fps.append(int(test_results[0]['test/fps']))
        fns.append(int(test_results[0]['test/fns']))
        ps.append(int(test_results[0]['test/ps']))
        ns.append(int(test_results[0]['test/ns']))
        write_session(out_file, 0, test_results, (fps,fns,ps,ns), 
                        data_module.get_class_balance(), train_len, None, None)

if __name__=='__main__':
    import time
    start_time = time.time()
    main()
    time_elapsed = time.time()-start_time
    print(time_elapsed)