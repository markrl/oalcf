import torch
import numpy as np
import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.model import CompClassModel
from unsupervised.module_unsup import VtdModule
from unsupervised.dataset_unsup import VtdDataModule
from unsupervised.params_unsup import get_params
from utils.utils import update_xent

from pdb import set_trace

def main():
    # Set pytorch precision
    torch.set_float32_matmul_precision('medium')
    # Get and handle parameters
    params = get_params()
    if params.overfit_batches >= 1:
        params.overfit_batches = int(params.overfit_batches)
    # Set up output directory
    out_dir = os.path.join('unsupervised/output', params.ckpt_name)
    if os.path.exists(out_dir):
        os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    # Save command
    command = ' '.join([os.path.basename(sys.executable)] + sys.argv)
    with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
        f.write(command)
    
    # Set up trainer callbacks
    callbacks = []
    ckpt_dir = 'checkpoints/'+params.ckpt_name
    if os.path.exists(ckpt_dir):
        os.system(f'rm -rf {ckpt_dir}')
    os.mkdir(ckpt_dir)
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best',
        monitor='val/loss',
        mode=params.mode
    ))
    callbacks.append(EarlyStopping(
        monitor='val/loss',
        mode=params.mode,
        patience=params.patience,
        min_delta=params.min_delta
    ))
    
    # Initialize lightning data module and lightning module
    data_module = VtdDataModule(params)
    module = VtdModule(params)
    # Instantiate trainer
    trainer = Trainer(
        callbacks=callbacks,
        fast_dev_run=params.debug,
        accelerator='gpu' if params.gpus>0 else 'cpu',
        devices=[0],
        overfit_batches=params.overfit_batches,
        max_epochs=params.max_epochs,
        check_val_every_n_epoch=params.val_every_n_epochs,
        logger=None,
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
    
    # Save final model
    new_state_dict = module.model.state_dict()
    out_state_dict = CompClassModel(params).state_dict()
    for kk in new_state_dict:
        if 'dec' not in kk:
            assert new_state_dict[kk].shape==out_state_dict[kk].shape
            out_state_dict[kk] = new_state_dict[kk]
    if not params.debug:
        torch.save(out_state_dict, os.path.join(out_dir, 'state_dict.pt'))

if __name__=='__main__':
    import time
    start_time = time.time()
    main()
    time_elapsed = time.time()-start_time
    print(time_elapsed)