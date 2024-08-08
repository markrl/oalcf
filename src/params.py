import argparse

def get_params():
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument('--run_name', type=str,
                        help='name of run')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--nworkers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='base learning rate')
    parser.add_argument('--val_every_n_epochs', type=int, default=1,
                        help='check validation set every n epochs')
    parser.add_argument('--gpus', type=int, default=1,
                        help='train on n gpus')
    parser.add_argument('--max_epochs', type=int, default=-1,
                        help='max number of epochs')
    parser.add_argument('--overfit_batches', type=float, default=0.0,
                        help='overfit batches')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='debug flag')
    parser.add_argument('--test_val', default=False, action='store_true',
                        help='train and validate on validation set')
    parser.add_argument('--monitor', type=str, default='train/loss',
                        help='metric to monitor for callbacks')
    parser.add_argument('--mode', type=str, default='min',
                        help='min or max')
    parser.add_argument('--patience', type=int, default=15,
                        help='patience for callbacks')
    parser.add_argument('--min_delta', type=int, default=0,
                        help='tolerance for callbacks')
    parser.add_argument('--seed', type=int, default=18792,
                        help='random seed')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--drop', type=float, default=0.0,
                        help='dropout rate')
    parser.add_argument('--schedule_lr', default=False, action='store_true',
                        help='use lr scheduler')
    parser.add_argument('--lrs_patience', type=int, default=10,
                        help='lr scheduler patience')
    parser.add_argument('--xent_weight', type=float, default=0.09,
                        help='cross-entropy loss weight for multi-loss model')
    parser.add_argument('--target_weight', type=float, default=4.0,
                        help='loss weighting for target class')
    parser.add_argument('--class_loss', type=str, default='xent',
                        help='classification loss type: `xent` or `dcf`')
    parser.add_argument('--dsmax_mult', type=float, default=0.0,
                        help='multiplier for differentiable softmax in DCF loss')
    parser.add_argument('--learn_mult', default=False, action='store_true',
                        help='learn dsmax multiplier')
    parser.add_argument('--learn_error_weight', default=False, action='store_true',
                        help='learn the error weighting for DCF loss')
    parser.add_argument('--auto_weight', default=False, action='store_true',
                        help='automatically change cross-entropy weighting based on training distribution')
    parser.add_argument('--auto_mult', type=float, default=1.0,
                        help='target cost for inverse weighting')
    parser.add_argument('--cb_loss', default=False, action='store_true',
                        help='use class-balanced loss')
    parser.add_argument('--beta', type=float, default=0.999,
                        help='factor for cb loss')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='focus parameter for focal loss')
    parser.add_argument('--load_pretrained', type=str, default=None,
                        help='path to pretrained model state dictionary')

    # OAL-CF arguments
    parser.add_argument('--al_methods', type=str, default='smax',
                        help='''list of active learning methods applied (separated by commas). 
                            `None` indicates no active learning.
                            options: `rand,ent,smax,necs`''')
    parser.add_argument('--n_queries', type=int, default=8,
                        help='number of active learning queries allotted per pass/batch')
    parser.add_argument('--epochs_per_pass', type=int, default=None,
                        help='set number of epochs per pass')
    parser.add_argument('--bootstrap', type=int, default=8,
                        help='number of bootstrap initialization samples')
    parser.add_argument('--boot_in', default=False, action='store_true',
                        help='extract bootstrap corpus from evaluation samples')
    parser.add_argument('--reset_weights', default=False, action='store_true',
                        help='reset model weights every pass')
    parser.add_argument('--combo', type=str, default=None,
                        help='combination method: `rank`, `plateau`, `split`, `rand` or None')
    parser.add_argument('--thresh', type=float, default=0.9,
                        help='threshold for the plateau combo method')
    parser.add_argument('--ddm', type=str, default=None,
                        help='drift detection method type; `nn`, `nc` or None')
    parser.add_argument('--ddm_thresh', type=float, default=3.1,
                        help='ddm threshold for continued sampling')
    parser.add_argument('--ddm_reduction', type=str, default='mean',
                        help='ddm reduction method (statistic)')
    parser.add_argument('--ddm_dist_fn', type=str, default=None,
                        help='ddm distance function')
    parser.add_argument('--drift_mult', type=int, default=5,
                        help='multiplicative factor for queries in drifting batches')
    parser.add_argument('--ddm_usage', type=str, default='mult',
                        help='how the ddm is applied; `thresh`, `stats`, or `mult`')
    parser.add_argument('--ddm_patience', type=int, default=10,
                        help='patience for drift reduction')
    parser.add_argument('--drift_budget', type=str, default=None,
                        help='ddm for drift-based dynamic budget allocation; `adwin` or `hdddm`')
    parser.add_argument('--budget_path', type=str, default=None,
                        help='load a predefined budget from this directory')
    parser.add_argument('--adapt_distr', type=float, default=None,
                        help='desired target class representation in adaptation data')
    parser.add_argument('--separate_class_al', default=False, action='store_true',
                        help='select AL samples for target and nontarget classes separately')
    parser.add_argument('--min_al_samples', type=int, default=2,
                        help='minimum number of AL samples per batch')
    parser.add_argument('--load_best', default=False, action='store_true',
                        help='use the best model from the previous session')
    parser.add_argument('--sim_type', type=str, default=None,
                        help='how to simulate CF')
    parser.add_argument('--max_fb_samples', type=int, default=None,
                        help='maximum number of CF samples per batch; `None` means no limit')
    parser.add_argument('--rank_cf', type=str, default=None,
                        help='choose the most informative feedback samples')
    parser.add_argument('--forget_n_batches', type=int, default=None,
                        help='forget adaptation data after n batches; `1`')
    parser.add_argument('--reverse_order', default=False, action='store_true',
                        help='run OAL backward')

    # Model arguments
    parser.add_argument('--no_initial_bn', default=False, action='store_true',
                        help='turn off initial batch norm layer')
    parser.add_argument('--comp_hidden', type=int, default=1024,
                        help='hidden layer size')
    parser.add_argument('--comp_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--class_hidden', type=int, default=1024,
                        help='classification network hidden dimension')
    parser.add_argument('--class_layers', type=int, default=3,
                        help='classification network layers')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='output embedding dimension')
    parser.add_argument('--rnn_layers', type=int, default=3,
                        help='number of recurrent layers')
    parser.add_argument('--context', type=int, default=3,
                        help='context frames on both sides of the target frame')
    parser.add_argument('--fill_gaps', default=False, action='store_true',
                        help='apply gap filling')
    parser.add_argument('--ensemble', default=False, action='store_true',
                        help='use ensemble model')

    # Data arguments
    parser.add_argument('--feat_root', type=str, default='/data/VTD/wavlm_11k_1hr/,/data/VTD/xvectors_11k_1hr',
                        help='root directory for features')
    parser.add_argument('--ann_root', type=str, default='/data/VTD/VTD_labels/all_labels_1hr',
                        help='path to annotations')
    parser.add_argument('--order_file', type=str, default='cluster',
                        help='path to file defining file ordering')
    parser.add_argument('--samples_per_batch', type=int, default=720,
                        help='number of samples per batch/session')
    parser.add_argument('--env_name', type=str, default='rm1_mc20',
                        help='environment specification')
    parser.add_argument('--lid_target', type=str, default='ha',
                        help='target class for LID task')


    return parser.parse_args()