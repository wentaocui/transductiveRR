import argparse
import time

def parser_args():
    parser = argparse.ArgumentParser(description='LaplacianSHot')
    ### dataset
    parser.add_argument('--data', metavar='DIR', default='./data/mini_imagenet/', help='path to dataset')
    parser.add_argument('--num-classes', type=int, default=64,
                        help='use all data to train the network')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--disable-train-augment', action='store_true',
                        help='disable training augmentation')
    parser.add_argument('--disable-random-resize', action='store_true',
                        help='disable random resizing')
    parser.add_argument('--jitter', default=True, type=bool, help='Image jitter added')
    parser.add_argument('--enlarge', action='store_false', help='enlarge the image size then center crop')
    ### network setting
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        help='network architecture')
    parser.add_argument('--scheduler', default='step', choices=('step', 'multi_step', 'cosine'),
                        help='scheduler, the detail is shown in train.py')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-stepsize', default=30, type=int,
                        help='learning rate decay step size ("step" scheduler) (default: 30)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='gamma for learning rate decay (default: 0.1)')
    parser.add_argument('--optimizer', default='SGD', choices=('SGD', 'Adam'))
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', action='store_true',
                        help='use nesterov for SGD, disable it in default')
    ### meta val setting
    parser.add_argument('--meta-test-iter', type=int, default=10000,
                        help='number of iterations for meta test')
    parser.add_argument('--meta-val-iter', type=int, default=500,
                        help='number of iterations for meta val')
    parser.add_argument('--meta-val-way', type=int, default=5,
                        help='number of ways for meta val/test')
    parser.add_argument('--meta-val-shot', type=int, default=1,
                        help='number of shots for meta val/test')
    parser.add_argument('--meta-val-query', type=int, default=15,
                        help='number of queries for meta val/test')
    parser.add_argument('--meta-val-interval', type=int, default=1,
                        help='do mate val in every D epochs')
    parser.add_argument('--meta-val-metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='euclidean',
                        help='meta-val/test evaluate metric')
    parser.add_argument('--num_NN', type=int, default=1,
                        help='number of nearest neighbors, set this number >1 when do kNN')
    parser.add_argument('--eval_fc', action='store_true',
                        help='do evaluate with final fc layer.')
    ### meta train setting
    parser.add_argument('--do-meta-train', action='store_true',
                        help='do prototypical training')
    parser.add_argument('--meta-train-iter', type=int, default=100,
                        help='number of iterations for meta val')
    parser.add_argument('--meta-train-way', type=int, default=5,
                        help='number of ways for meta val')
    parser.add_argument('--meta-train-shot', type=int, default=1,
                        help='number of shots for meta val')
    parser.add_argument('--meta-train-query', type=int, default=15,
                        help='number of queries for meta val')
    parser.add_argument('--meta-train-metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='euclidean',
                        help='meta-train evaluate metric')
    ### others
    parser.add_argument('--split-dir', default='./split/mini', type=str,
                        help='path to the folder stored split files.')
    parser.add_argument('--save-path', default='./results/mini/softmax/resnet18', type=str,
                        help='path to folder stored the log and checkpoint')
    parser.add_argument('--log-file', default='/training.log', type=str,
                        help='log-file')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--disable-tqdm', action='store_true',
                        help='disable tqdm.')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--cutmix_prob', default=0, type=float,
                        help='cutmix probability. Open cutmix augmentation when cutmix_prob>0 and beta >0')
    parser.add_argument('--beta', default=-1., type=float,
                        help='cutmix beta. Open cutmix augmentation when cutmix_prob>0 and beta >0')
    parser.add_argument('--evaluate', action='store_false',
                        help='evaluate the final result')
    parser.add_argument('--pretrain', type=str, default='./results/mini/softmax/resnet18',
                        help='path to the pretrained model, used for fine-tuning')
    parser.add_argument('--label-smooth', type=float, default = 0.1,
                        help='Label smoothing. 0.0 means no label smoothing')

    ## LaplacianShot

    parser.add_argument('--lmd', default=1, type=float,
                        help='weight for Laplacian')
    parser.add_argument('--knn', default=3, type=int,
                        help='knn for affinity')
    parser.add_argument('--lshot', action='store_false',
                        help='enable LaplacianShot.')
    parser.add_argument('--tune-lmd', action='store_true',
                        help='Tune Lambda on Validation set.')
    parser.add_argument('--proto-rect', default = True,  type=str2bool,
                        help='Prototype rectification')
    parser.add_argument('--plot-converge', default = False,  type=str2bool,
                        help='plot the energy in each bound updates.')

    parser.add_argument('--CUDA-VISIBLE-DEVICES', type=str, default='0')
    parser.add_argument('--enable-transduct', action='store_true')
    parser.add_argument('--result-file-idx', default=1, type=int)
    parser.add_argument('--eval-shot', default=1, type=int, help='')
    parser.add_argument('--eval-used-set', type=str, default='test')
    parser.add_argument('--transduct-alpha1', type=float, default=0)
    parser.add_argument('--transduct-alpha2', type=float, default=0)
    parser.add_argument('--transduct-tau', type=float, default=4)
    parser.add_argument('--root-path', type=str, default='./')
    parser.add_argument('--data-for-all', type=str, default='miniImagenet')
    parser.add_argument('--arch-for-all', type=str, default='resnet18')

    parser.add_argument('--auto-transduct', action='store_false', help='')

    args = parser.parse_args()

    if args.auto_transduct == True:
        args = load_transduct_params(args)

    return args

def load_transduct_params(args):
    if args.meta_val_way == 5:
        if args.data_for_all == 'miniImagenet':
            if args.arch_for_all == 'resnet18':
                if args.eval_shot == 1:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.7, 0.9, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.1, 0.5, 4
            elif args.arch_for_all == 'wideres':
                if args.eval_shot == 1:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.7, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.1, 0.5, 4
        elif args.data_for_all == 'tiered_imagenet':
            if args.arch_for_all == 'resnet18':
                if args.eval_shot == 1:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.8, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.1, 0.2, 4
            elif args.arch_for_all == 'wideres':
                if args.eval_shot == 1:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.4, 0.7, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.1, 0.5, 4
        elif args.data_for_all == 'cub':
            if args.arch_for_all == 'resnet18':
                if args.eval_shot == 1:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.8, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.5, 4
            elif args.arch_for_all == 'wideres':
                if args.eval_shot == 1:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.6, 0.9, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.7, 4
    elif args.meta_val_way in [10,20] and args.data_for_all == 'miniImagenet' and args.arch_for_all == 'resnet18':
        if args.eval_shot == 1:
            args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.7, 0.9, 4
        elif args.eval_shot == 5:
            args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.1, 0.6, 4

    print('%s  %s  LaplacianShot  test_way %d  shot %d  enable_transduct %s  alpha1 %s  alpha2 %s  tau %s' %
          (args.data_for_all, args.arch_for_all, args.meta_val_way, args.eval_shot, args.enable_transduct, args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau))
    return args




def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def write_log(params, shot, accuracy_info, best_lmd):
    with open(params.root_path + '/results_%d.txt' %(params.result_file_idx), 'a') as f:
        timestamp = time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime())
        exp_setting = '%s--%s--%sway--%sshot--%s--a1 %s--a2 %s--tau %s--transduct %s' %(params.split_dir.split('/')[-2], params.arch,
                                                                      params.meta_val_way, shot, params.eval_used_set,
                                                                      params.transduct_alpha1, params.transduct_alpha2, params.transduct_tau, params.enable_transduct)
        acc = '{:.2f} +- {:.2f} | {:.2f} +- {:.2f} | {:.2f} +- {:.2f} | {:.2f}'.format(*accuracy_info, best_lmd)
        f.write( '%s | %s | %s \n' %(timestamp, exp_setting, acc)  )


def set_args_shortcut(args):
    if args.data_for_all == 'miniImagenet':
        args.data = './data/mini_imagenet/'
        args.num_classes = 64
        args.split_dir = './split/mini/'
        if args.arch_for_all == 'resnet18':
            args.arch = 'resnet18'
            args.save_path = './results/mini/softmax/resnet18'
            if args.eval_shot == 1:
                args.lmd = 0.7
            elif args.eval_shot == 5:
                args.lmd = 0.1
            else:
                raise ValueError('only 1 shot and 5 shot')
        elif args.arch_for_all == 'wideres':
            args.arch = 'wideres'
            args.save_path = './results/mini/softmax/wideres'
            if args.eval_shot == 1:
                args.lmd = 0.5
            elif args.eval_shot == 5:
                args.lmd = 0.3
            else:
                raise ValueError('only 1 shot and 5 shot')
        else:
            raise ValueError('only resnet18 and wideres can be arch_for_all shortcut')
        args.pretrain = args.save_path
    elif args.data_for_all == 'tiered_imagenet':
        args.data = './data/tiered_imagenet/data/'
        args.num_classes = 351
        args.split_dir = './split/tiered/'
        if args.arch_for_all == 'resnet18':
            args.arch = 'resnet18'
            args.save_path = './results/tiered/softmax/resnet18'
            if args.eval_shot == 1:
                args.lmd = 0.7
            elif args.eval_shot == 5:
                args.lmd = 0.1
            else:
                raise ValueError('only 1 shot and 5 shot')
        elif args.arch_for_all == 'wideres':
            args.arch = 'wideres'
            args.save_path = './results/tiered/softmax/wideres'
            if args.eval_shot == 1:
                args.lmd = 0.5
            elif args.eval_shot == 5:
                args.lmd = 0.1
            else:
                raise ValueError('only 1 shot and 5 shot')
        else:
            raise ValueError('only resnet18 and wideres can be arch_for_all shortcut')
        args.pretrain = args.save_path
    elif args.data_for_all == 'cub':
        args.data = './data/cub/CUB_200_2011/images/'
        args.num_classes = 100
        args.split_dir = './split/cub/'
        if args.arch_for_all == 'resnet18':
            args.arch = 'resnet18'
            args.save_path = './results/cub/softmax/resnet18'
            if args.eval_shot == 1:
                args.lmd = 0.7
            elif args.eval_shot == 5:
                args.lmd = 0.1
            else:
                raise ValueError('only 1 shot and 5 shot')
        elif args.arch_for_all == 'wideres':
            args.arch = 'wideres'
            args.save_path = './results/cub/softmax/wideres'
            if args.evaluate == False:
                # set base training related hyperparameter
                args.batch_size = 40
                args.meta_val_interval = 4
                args.meta_val_metric = 'cosine'
                args.scheduler = 'multi_step'
                args.lshot = False
                args.proto_rect = False
                print('batch_size_{} | meta_eval_interval_{} | meta_val_metric_{} | scheduler_{} | lshot_{} | proto_rect_{}'.format(
                    args.batch_size, args.meta_val_interval, args.meta_val_metric, args.scheduler, args.lshot, args.proto_rect))
            if args.eval_shot == 1:
                args.lmd = 0.7
            elif args.eval_shot == 5:
                args.lmd = 0.1
            else:
                raise ValueError('only 1 shot and 5 shot')
        else:
            raise ValueError('only resnet18 can be arch_for_all shortcut for cub')
        args.pretrain = args.save_path
    else:
        raise ValueError('only mini, tiered, cub can be data_for_all shortcut')

    return args