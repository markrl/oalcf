import torch
import os
import sys
from copy import deepcopy

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

from src.module import VtdModule
from src.dataset import VtdImlDataModule
from src.params import get_params
from utils.query_strategies import StrategyManager
from utils.utils import reset_trainer, update_xent
from utils.utils import write_header, write_session
from utils.ddm import NcDdm, NnDdm

from pdb import set_trace

def main():
    # Get parameters
    params = get_params()
    sm = StrategyManager(params)
    if params.ddm is None:
        ddm = None
    else:
        if params.ddm=='nc':
            ddm = NcDdm(n_clusters=params.n_queries, reduction=params.ddm_reduction, dist_fn=params.ddm_dist_fn)
        else:
            ddm = NnDdm(reduction=params.ddm_reduction, dist_fn=params.ddm_dist_fn)
    if params.overfit_batches >= 1:
        params.overfit_batches = int(params.overfit_batches)
    # Set up directory
    out_dir = os.path.join('output', params.run_name)
    if os.path.exists(out_dir):
        os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    # Save command
    command = ' '.join([os.path.basename(sys.executable)] + sys.argv)
    with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
        f.write(command)
    # Set pytorch precision
    torch.set_float32_matmul_precision('medium')
    # Instantiate trainer
    callbacks = []
    ckpt_dir = 'checkpoints/'+params.run_name
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
    
    data_module = VtdImlDataModule(params)
    module = VtdModule(params)
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

    # Set up variables and output file
    if params.al_methods[-1] == ',':
        params.al_methods = params.al_methods[:-1]
    al_methods = params.al_methods.split(',')
    al_methods.sort()
    if not params.debug:
        out_file = os.path.join(out_dir, 'scores.csv')
        write_header(out_file, al_methods)

    # Form bootstrap corpus
    data_module.label_boot()
    if params.reset_weights:
        base_state_dict = module.model.state_dict()
    else:
        base_state_dict = None
    data_module.next_batch()
    if params.auto_weight and params.class_loss=='xent':
        update_xent(module, data_module)
    trainer.fit(module, data_module)

    # Move into OAL training
    ps, ns = [], []
    fps, fns = [], []
    mm = params.combo if params.combo is not None else al_methods[0]
    while data_module.current_batch <= data_module.n_batches:
        print(f'STARTING {data_module.get_current_session_name()} ({data_module.current_batch}/{data_module.n_batches})')
        idxs_dict, metrics_dict = sm.select_queries(data_module, al_methods, module, params.n_queries)
        if base_state_dict is not None:
            module.model.load_state_dict(base_state_dict)
        idxs = idxs_dict[mm]
        data_module.transfer_samples(idxs)
        reset_trainer(trainer)
        if params.auto_weight and params.class_loss=='xent':
            update_xent(module, data_module)
        trainer.fit(module, data_module)
        if not params.debug and params.load_best:
            module = VtdModule.load_from_checkpoint(ckpt_dir+'/best.ckpt')
        os.system(f'rm -rf {ckpt_dir}') # Delete old checkpoints
        test_results = trainer.test(module, data_module)
        if not params.debug:
            write_session(out_file, data_module.current_batch, test_results)
        data_module.next_batch()
    if not params.debug:
        torch.save(module.model.state_dict(), os.path.join(out_dir, 'state_dict.pt'))
        data_module.save_active_files(os.path.join(out_dir, 'al_samples.txt'))

if __name__=='__main__':
    import time
    start_time = time.time()
    main()
    time_elapsed = time.time()-start_time
    print(time_elapsed)