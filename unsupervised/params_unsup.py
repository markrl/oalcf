import argparse

def get_params():
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument('--ckpt_name', type=str,
                        help='model checkpoint name')
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
    parser.add_argument('--patience', type=int, default=3,
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
    parser.add_argument('--lrs_patience', type=int, default=2,
                        help='lr scheduler patience')
    parser.add_argument('--xent_weight', type=float, default=0.09,
                        help='cross-entropy loss weight for multi-loss model')
    parser.add_argument('--target_weight', type=float, default=4.0,
                        help='loss weighting for target class')
    parser.add_argument('--class_loss', type=str, default='xent',
                        help='classification loss type: `xent` or `dcf`')
    parser.add_argument('--dsmax_mult', type=float, default=0.0,
                        help='multiplier for differentiable softmax in DCF loss')
    parser.add_argument('--auto_weight', default=False, action='store_true',
                        help='automatically change cross-entropy weighting based on training distribution')

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

    # Data arguments
    parser.add_argument('--feat_root', type=str, default='/data/VTD/wavlm_11k_1hr/,/data/VTD/xvectors_11k_1hr',
                        help='root directory for features')
    parser.add_argument('--ann_root', type=str, default='/data/VTD/VTD_labels/all_labels_1hr',
                        help='path to annotations')
    parser.add_argument('--samples_per_batch', type=int, default=720,
                        help='number of samples per batch/session')
    parser.add_argument('--corpus', type=str, default='sri',
                        help='corpus specification')
    parser.add_argument('--env_name', type=str,
                        help='environment details; [room]_[mic]')

    return parser.parse_args()