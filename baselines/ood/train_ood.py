import torch
import numpy as np
import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.module import VtdModule
from baselines.ood.dataset_ood import SupervisedDataModule
from baselines.ood.params_ood import get_params
from utils.utils import update_xent
from utils.utils import write_header, write_session

from pdb import set_trace

def main():
    # Set pytorch precision
    torch.set_float32_matmul_precision('medium')
    # Get and handle parameters
    params = get_params()
    if params.overfit_batches >= 1:
        params.overfit_batches = int(params.overfit_batches)
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