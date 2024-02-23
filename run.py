import torch
import numpy as np
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
    # Set pytorch precision
    torch.set_float32_matmul_precision('medium')
    # Get and handle parameters
    params = get_params()
    if params.overfit_batches >= 1:
        params.overfit_batches = int(params.overfit_batches)
    # Initialize query selection strategy object
    sm = StrategyManager(params)
    # Initialize DDM
    if params.ddm is None:
        ddm = None
    else:
        drift_list = []
        if params.ddm=='nc':
            ddm = NcDdm(n_clusters=params.n_queries, reduction=params.ddm_reduction, dist_fn=params.ddm_dist_fn)
        else:
            ddm = NnDdm(reduction=params.ddm_reduction, dist_fn=params.ddm_dist_fn)
    # Set up output directory
    out_dir = os.path.join('output', params.run_name)
    if os.path.exists(out_dir):
        os.system(f'rm -rf {out_dir}')
    os.mkdir(out_dir)
    # Save command
    command = ' '.join([os.path.basename(sys.executable)] + sys.argv)
    with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
        f.write(command)
    # Set up variables and output file
    if params.al_methods[-1] == ',':
        params.al_methods = params.al_methods[:-1]
    al_methods = params.al_methods.split(',')
    al_methods.sort()
    if not params.debug:
        out_file = os.path.join(out_dir, 'scores.csv')
        write_header(out_file, al_methods, (ddm is not None))
    
    # Set up trainer callbacks
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
    
    # Initialize lightning data module and lightning module
    data_module = VtdImlDataModule(params)
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

    # Prepare base model if resetting weights every batch
    if params.reset_weights:
        base_state_dict = module.model.state_dict()
    else:
        base_state_dict = None
    
    # Form bootstrap corpus
    data_module.label_boot()
    data_module.next_batch()
    module.n_train = len(data_module)
    # Update class weights if indicated
    if params.auto_weight and params.class_loss=='xent':
        update_xent(module, data_module)
    # Delete old checkpoints
    os.system(f'rm -rf {ckpt_dir}/best.ckpt') 
    # Train on bootstrap corpus
    trainer.fit(module, data_module)

    ### IML LOOP ###
    # Initialize metric tracking lists
    ps, ns = [], []
    fps, fns = [], []
    pre_fps, pre_fns = [], []
    budget = 0
    mm = 'combo' if params.combo is not None else al_methods[0]
    # Loop through all batches
    while data_module.current_batch < data_module.n_batches:
        # Print current session name and number
        print(f'STARTING {data_module.get_current_session_name()} ({data_module.current_batch+1}/{data_module.n_batches})')
        # Get prequential evaluation metrics from this batch
        test_results = trainer.test(module, data_module)
        # Save results
        if not params.debug:
            pre_fps.append(int(test_results[0]['test/fps']))
            pre_fns.append(int(test_results[0]['test/fns']))

        # Handle DDM, budgeting, query selection, etc. (TODO: Refactor and make cleaner)
        if params.budget_path is not None:
            n_queries = int(np.genfromtxt(os.path.join(params.budget_path, params.env_name, 'budget.txt'))[data_module.current_batch])
        else:
            n_queries = params.n_queries

        if ddm is not None:
            dist = ddm.get_dist(data_module.data_train, data_module.data_test)
            if params.ddm_usage=='mult':
                mult = params.drift_mult if dist>params.ddm_thresh else 1
                idxs_dict, metrics_dict = sm.select_queries(data_module, al_methods, module, n_queries*mult)
                data_module.transfer_samples(idxs_dict[mm])
            elif params.ddm_usage=='thresh':
                min_dist = dist
                wait_time = 0
                metrics_dict = {mm:0}
                while dist>params.ddm_thresh and len(data_module.data_test)>680 and wait_time<params.ddm_patience:
                    idxs_dict, metrics_dict = sm.select_queries(data_module, al_methods, module, n_queries)
                    data_module.transfer_samples(idxs_dict[mm])
                    dist = ddm.get_dist(data_module.data_train, data_module.data_test)
                    wait_time += 1
                    if dist < min_dist:
                        min_dist = dist
                        wait_time = 0
            elif params.ddm_usage=='stats':
                if len(drift_list)>0:
                    drift_mean = torch.mean(torch.tensor(drift_list))
                    mult = dist/drift_mean
                    n_queries = int(n_queries*mult)
                drift_list.append(dist)
                idxs_dict, metrics_dict = sm.select_queries(data_module, al_methods, module, n_queries)
                data_module.transfer_samples(idxs_dict[mm])
            else:
                exit()
        else:
            if params.separate_class_al:
                dist = None
                budget += n_queries
                sm.est_class = 'target'
                sm.min_samples = params.min_al_samples
                idxs_dict, metrics_dict = sm.select_queries(data_module, al_methods, module, budget)
                data_module.transfer_samples(idxs_dict[mm])
                print(f'Target queries: {len(idxs_dict[mm])}')
                budget -= len(idxs_dict[mm])
                sm.est_class = 'nontarget'
                sm.min_samples = np.maximum(0, sm.min_samples-len(idxs_dict[mm]))
                idxs_dict, metrics_dict = sm.select_queries(data_module, al_methods, module, budget)
                data_module.transfer_samples(idxs_dict[mm])
                print(f'Nontarget queries: {len(idxs_dict[mm])}')
                p_t, p_n = data_module.get_class_balance()
                print(f'P_t: {p_t:.4f}')
                budget -= len(idxs_dict[mm])
                print(f'Remaining budget: {budget}')
            else:
                dist = None
                idxs_dict, metrics_dict = sm.select_queries(data_module, al_methods, module, n_queries)
                data_module.transfer_samples(idxs_dict[mm])

        # Handle exception where a batch only contains 1 sample
        data_module.drop_last = True if len(data_module)%params.batch_size==1 else False
        # Reset trainer for the new batch
        reset_trainer(trainer)
        # Reload base model if resetting every batch
        if base_state_dict is not None:
            module.model.load_state_dict(base_state_dict)
        # Update class weighting if indicated
        if params.auto_weight and params.class_loss=='xent':
            update_xent(module, data_module)
        # Delete old checkpoints
        os.system(f'rm -rf {ckpt_dir}/best.ckpt')
        # Train model on adaptation pool
        trainer.fit(module, data_module)
        # Load model with from best epoch for this batch if indicated
        if not params.debug and params.load_best:
            module = VtdModule.load_from_checkpoint(ckpt_dir+'/best.ckpt')
        # Get postquential evaluation metrics from this batch
        test_results = trainer.test(module, data_module)
        # Write results to file
        if not params.debug:
            fps.append(int(test_results[0]['test/fps']))
            fns.append(int(test_results[0]['test/fns']))
            ps.append(int(test_results[0]['test/ps']))
            ns.append(int(test_results[0]['test/ns']))
            metric = None if len(al_methods)>1 or mm=='rand' else metrics_dict[mm]
            write_session(out_file, data_module.current_batch, test_results, (pre_fps,pre_fns,fps,fns,ps,ns), 
                            data_module.get_class_balance(), len(data_module.data_train), metric, dist)
        # Load next batch
        data_module.next_batch()
        module.n_train = len(data_module)
    # Save final model and AL samples
    if not params.debug:
        torch.save(module.model.state_dict(), os.path.join(out_dir, 'state_dict.pt'))
        data_module.save_active_files(os.path.join(out_dir, 'al_samples.txt'))

if __name__=='__main__':
    import time
    start_time = time.time()
    main()
    time_elapsed = time.time()-start_time
    print(time_elapsed)