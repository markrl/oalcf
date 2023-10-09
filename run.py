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
        num_sanity_val_steps=1 if params.debug else 0
    )

    if not params.debug:
        out_file = os.path.join(out_dir, 'scores.csv')
        f = open(out_file, 'w')

    # Set up variables and output file
    best_n_samples, best_p_nontarget, best_p_target = 0, 0, 0
    best_score = 1
    if params.al_methods[-1] == ',':
        params.al_methods = params.al_methods[:-1]
    al_methods = params.al_methods.split(',')
    al_methods.sort()
    if not params.debug:
        if al_methods[0]=='rand' or len(al_methods)>1:
            f.write('pass,auprc,dcf,fnr,fpr,ns,ps,fns,fps,p_target,p_nontarget,n_samples,cum_dcf\n')
        else:
            f.write('pass,auprc,dcf,fnr,fpr,ns,ps,fns,fps,p_target,p_nontarget,n_samples,cum_dcf,metric\n')

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
    mm = params.combo if params.combo is not None else params.al_methods
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
        if not params.debug:
            module = VtdModule.load_from_checkpoint(ckpt_dir+'/best.ckpt')
        test_results = trainer.test(module, data_module)
        if not params.debug:
            f.write(f'{data_module.current_batch-1}')
            auprc = test_results[0]['test/auprc']
            dcf = test_results[0]['test/dcf']
            fnr = test_results[0]['test/fnr']
            fpr = test_results[0]['test/fpr']
            f.write(f',{auprc:.4f},{dcf:.4f},{fnr:.4f},{fpr:.4f}')
            fps.append(int(test_results[0]['test/fps']))
            fns.append(int(test_results[0]['test/fns']))
            ps.append(int(test_results[0]['test/ps']))
            ns.append(int(test_results[0]['test/ns']))
            f.write(f',{ns[-1]:d},{ps[-1]:d},{fns[-1]:d},{fps[-1]:d}')
            p_target,p_nontarget = data_module.get_class_balance()
            n_samples = len(data_module.data_train)
            f.write(f',{p_target:.4f},{p_nontarget:.4f},{n_samples:d}')
            fnr = torch.sum(torch.LongTensor(fns))/torch.sum(torch.LongTensor(ps)) if torch.sum(torch.LongTensor(ps))>0 else 0
            fpr = torch.sum(torch.LongTensor(fps))/torch.sum(torch.LongTensor(ns)) if torch.sum(torch.LongTensor(ns))>0 else 0
            cum_dcf = 0.75*fnr + 0.25*fpr
            f.write(f',{cum_dcf:.4f}')
            if al_methods[0]=='rand' or len(al_methods)>1:
                f.write('\n')
            else:
                f.write(f',{metrics_dict[mm]:.4f}\n')
        data_module.next_batch()
    if not params.debug:
        torch.save(module.model.state_dict(), os.path.join(out_dir, 'state_dict.pt'))
        data_module.save_active_files(os.path.join(out_dir, 'al_samples.txt'))
    f.close()

if __name__=='__main__':
    import time
    start_time = time.time()
    main()
    time_elapsed = time.time()-start_time
    print(time_elapsed)