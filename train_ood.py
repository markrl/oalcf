import torch
import numpy as np
import os
import sys
import glob
import pickle

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.module import VtdModule
from baselines.ood.dataset_ood import SupervisedDataModule
from baselines.ood.params_ood import get_params
from utils.utils import update_xent
from utils.utils import write_header, write_session

from pdb import set_trace

def main():
    # General pytorch settings
    torch.set_flush_denormal(True)
    torch.set_num_threads(1)
    # Get and handle parameters
    params = get_params()
    # Set pytorch precision
    torch.set_float32_matmul_precision('high')
    # Determine whether to use a GPU
    use_gpu = (params.gpus>0 and torch.cuda.is_available())
    if not use_gpu:
        torch.use_deterministic_algorithms(True)
    if params.overfit_batches >= 1:
        params.overfit_batches = int(params.overfit_batches)
    # Handle feature directory
    if params.feat_root=='auto':
        if os.path.exists(os.path.join(os.path.expanduser('~'), 'data')):
            data_root = os.path.join(os.path.expanduser('~'), 'data')
        else:
            data_root = '/data'
        if params.lid_target is None:
            params.feat_root = os.path.join(data_root, 'vtd/wavlm_11k_1hr')+','+os.path.join(data_root, 'vtd/xvectors_11k_1hr')
            params.ann_root = os.path.join(data_root, 'vtd/all_labels_1hr')
        else:
            for corpus in os.listdir(os.path.join(data_root, 'slv')):
                file_list = glob.glob(os.path.join(data_root, 'slv', corpus, 
                                                    'ecapalang', params.env_name, '*.npy'))
                langs_list = [os.path.basename(ff).split('_')[2] for ff in file_list]
                langs_list_unique = []
                for lang in langs_list:
                    if lang not in langs_list_unique:
                        langs_list_unique.append(lang)
                if params.lid_target in langs_list:
                    params.feat_root = os.path.join(data_root, 'slv', corpus, 'ecapalang')
    # Set up output directory
    out_dir = os.path.join('baselines/ood/output', params.ckpt_name)
    if os.path.exists(out_dir):
        os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    # Save command
    command = ' '.join([os.path.basename(sys.executable)] + sys.argv)
    with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
        f.write(command)
    # Set up variables and output file
    if not params.debug:
        out_file = os.path.join(out_dir, 'scores.csv')
        write_header(out_file, [], False)
    
    # Set up trainer callbacks
    callbacks = []
    ckpt_dir = 'checkpoints/'+params.ckpt_name
    if os.path.exists(ckpt_dir):
        os.system(f'rm -rf {ckpt_dir}')
    os.mkdir(ckpt_dir)
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best',
        monitor=params.monitor,
        mode=params.mode
    ))
    callbacks.append(EarlyStopping(
        monitor=params.monitor,
        mode=params.mode,
        patience=params.patience,
        min_delta=params.min_delta
    ))
    
    # Initialize lightning data module and lightning module
    data_module = SupervisedDataModule(params)
    module = VtdModule(params)
    # Instantiate trainer
    trainer = Trainer(
        callbacks=callbacks,
        fast_dev_run=params.debug,
        accelerator='gpu' if use_gpu else 'cpu',
        devices=[0] if use_gpu else 1,
        overfit_batches=params.overfit_batches,
        max_epochs=params.max_epochs,
        check_val_every_n_epoch=params.val_every_n_epochs,
        logger=False,
        log_every_n_steps=1,
        num_sanity_val_steps=1 if params.debug else 0
    )
    
    # Update class weights if indicated
    if params.auto_weight and params.class_loss=='xent':
        update_xent(module, data_module)
    # Delete old checkpoints
    os.system(f'rm -rf {ckpt_dir}')
    # Train on training corpus
    trainer.fit(module, data_module)
    # Evaluate on test corpus
    test_results = trainer.test(module, data_module)

    # Write results to file
    fps, fns, ps, ns = [], [], [], []
    if not params.debug:
        fps.append(int(test_results[0]['test/fps']))
        fns.append(int(test_results[0]['test/fns']))
        ps.append(int(test_results[0]['test/ps']))
        ns.append(int(test_results[0]['test/ns']))
        write_session(out_file, 0, test_results, (fps,fns,ps,ns,fps,fns,ps,ns), 
                        data_module.get_class_balance(), len(data_module.data_train)+len(data_module.data_val), 
                        None, None, 0, 0, 0, 0)
    
    # Save final model
    if not params.debug:
        if params.ensemble:
            pickle.dump(module.model.model, open('model.p', 'wb'))
        else:
            torch.save(module.model.state_dict(), os.path.join(out_dir, 'state_dict.pt'))

if __name__=='__main__':
    import time
    start_time = time.time()
    main()
    time_elapsed = time.time()-start_time
    print(time_elapsed)