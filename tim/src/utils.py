import torch
import numpy as np
import shutil
from tqdm import tqdm
import logging
import os
import pickle
import torch.nn.functional as F
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    # general parameter
    parser.add_argument('--root-path', default='./', type=str, help='')
    parser.add_argument('--ckpt-path',  default='./checkpoints/cub/softmax/wideres',    type=str,   help='')
    parser.add_argument('--log',        default='./logs/tim_gd/cub/',            type=str,   help='')
    parser.add_argument('--evaluate',   action='store_false',  help='')
    parser.add_argument('--enable-transduct',    action='store_true',    help='')

    parser.add_argument('--cuda-device', default=0,                                     type=int, help='')
    parser.add_argument('--seed',       default=2020,                                   type=int,   help='')
    parser.add_argument('--pretrain',   default=False,                                  type=bool,  help='')
    parser.add_argument('--resume',     default=False,                                  type=bool,  help='')
    parser.add_argument('--make-plot',  default=False,                                  type=bool,  help='')
    parser.add_argument('--epochs',     default=90,                                     type=int,   help='')
    parser.add_argument('--disable-tqdm', default=False,                                type=bool,  help='')
    parser.add_argument('--visdom-port', default=None,                                  type=int,   help='')
    parser.add_argument('--print-runtime', default=False,                               type=bool,  help='')
    parser.add_argument('--cuda',       default=True,                                   type=bool,  help='')

    # dataset parameter
    parser.add_argument('--dataset-path', default='./data/cub/CUB_200_2011/images',                 type=str,   help='')
    parser.add_argument('--dataset-split_dir', default='split/cub',                    type=str,   help='')

    parser.add_argument('--dataset-batch-size', default=256,                            type=int, help='')
    parser.add_argument('--dataset-enlarge', default=True,                              type=bool, help='')
    parser.add_argument('--dataset-num-workers', default=4,                             type=int, help='')
    parser.add_argument('--dataset-disable-random-resize', default=False,               type=bool, help='')
    parser.add_argument('--dataset-jitter', default=False,                              type=bool, help='')

    # model parameter
    parser.add_argument('--model-arch', default='resnet18',                             type=str,   help='')

    parser.add_argument('--model-num-classes', default=100,                              type=int,   help='')

    # tim parameter
    parser.add_argument('--tim-iter', default=1000,                                      type=int,   help='')

    parser.add_argument('--tim-temp', default=15,                                       type=int,   help='')
    parser.add_argument('--tim-loss-weights', default=[0.1, 1.0, 0.1],                              help='')
    parser.add_argument('--tim-lr', default=1e-4,                                       type=float, help='')
    parser.add_argument('--tim-alpha', default=1.0,                                     type=float, help='')

    # eval parameter
    parser.add_argument('--eval-method', default='tim_gd',                              type=str,   help='')

    parser.add_argument('--eval-target-data-path', default=None,                        type=str, help='')
    parser.add_argument('--eval-target-split-dir', default=None,                        type=str,   help='')

    parser.add_argument('--eval-number-tasks', default=10000,                           type=int,   help='')
    parser.add_argument('--eval-n-ways', default=5,                                     type=int,   help='')
    parser.add_argument('--eval-query-shots', default=15,                               type=int,   help='')
    parser.add_argument('--eval-model-tag', default='best',                             type=str,   help='')
    parser.add_argument('--eval-plt-metrics', default=['accs'],                                     help='')
    parser.add_argument('--eval-shot', default=1,                                       type=int,   help='')
    parser.add_argument('--eval-used-set', default='test',                              type=str,   help='')

    # trainer parameter
    parser.add_argument('--trainer-print-freq', default=10,                             type=int, help='')
    parser.add_argument('--trainer-meta-val-way', default=5,                            type=int, help='')
    parser.add_argument('--trainer-meta-val-shot', default=1,                           type=int, help='')
    parser.add_argument('--trainer-meta-val-metric', default='cosine',                  type=str, help='')
    parser.add_argument('--trainer-meta-val-iter', default=500,                         type=int, help='')
    parser.add_argument('--trainer-meta-val-query', default=15,                         type=int, help='')
    parser.add_argument('--trainer-alpha', default=-1.0,                                type=float, help='')
    parser.add_argument('--trainer-label-smoothing', default=0.,                        type=float, help='')

    # optim parameter
    parser.add_argument('--optim-gamma', default=0.1,                                   type=float, help='')
    parser.add_argument('--optim-lr', default=0.1,                                      type=float, help='')
    parser.add_argument('--optim-lr-stepsize', default=30,                              type=int, help='')
    parser.add_argument('--optim-nesterov', default=False,                              type=bool, help='')
    parser.add_argument('--optim-weight-decay', default=1e-4,                           type=float, help='')
    parser.add_argument('--optim-optimizer-name', default='SGD',                        type=str, help='')
    parser.add_argument('--optim-scheduler', default='multi_step',                      type=str, help='')

    # transduction
    parser.add_argument('--transduct-alpha1', default=0,                                type=float, help='')
    parser.add_argument('--transduct-alpha2', default=0,                                type=float, help='')
    parser.add_argument('--transduct-tau', default=4,                                type=float, help='')
    parser.add_argument('--CUDA-VISIBLE-DEVICES', type=str, default='0')
    parser.add_argument('--arch-for-all', default='resnet18', type=str, help='')
    parser.add_argument('--data-for-all', default='miniImagenet', type=str, help='')
    parser.add_argument('--result-file-idx', default=1, type=int)
    parser.add_argument('--iterative-transduct', action='store_true')
    parser.add_argument('--n-ierative-transduct', default=2, type=int)
    parser.add_argument('--auto-transduct', action='store_false', help='')

    args = parser.parse_args()

    if args.auto_transduct == True:
        args = load_transduct_params(args)

    return args


def load_transduct_params(args):
    if args.eval_n_ways == 5:
        if args.data_for_all == 'miniImagenet':
            if args.arch_for_all == 'resnet18':
                if args.eval_shot == 1:
                    if args.eval_query_shots == 5:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.5, 4
                    elif args.eval_query_shots == 10:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.4, 0.4, 4
                    elif args.eval_query_shots == 15:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.5, 4
                    elif args.eval_query_shots == 20:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.7, 0.7, 4
                    elif args.eval_query_shots == 25:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.8, 0.8, 4
                    elif args.eval_query_shots == 30:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.6, 0.6, 4
                if args.eval_shot == 2:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.4, 0.2, 4
                if args.eval_shot == 3:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.4, 4
                if args.eval_shot == 4:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.4, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.4, 0.4, 4
                if args.eval_shot == 6:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.2, 4
                if args.eval_shot == 7:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0, 4
                if args.eval_shot == 8:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.5, 4
                if args.eval_shot == 9:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.3, 4
                if args.eval_shot == 10:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.1, 0.1, 4
            elif args.arch_for_all == 'wideres':
                if args.eval_shot == 1:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.5, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.3, 4
        elif args.data_for_all == 'tiered_imagenet':
            if args.arch_for_all == 'resnet18':
                if args.eval_shot == 1:
                    if args.eval_query_shots == 5:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.4, 4
                    elif args.eval_query_shots == 10:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.3, 4
                    elif args.eval_query_shots == 15:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.5, 4
                    elif args.eval_query_shots == 20:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.5, 4
                    elif args.eval_query_shots == 25:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.3, 4
                    elif args.eval_query_shots == 30:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.6, 0.6, 4
                elif args.eval_shot == 2:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.4, 4
                elif args.eval_shot == 3:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.3, 4
                elif args.eval_shot == 4:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.5, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.2, 4
                elif args.eval_shot == 6:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.2, 4
                elif args.eval_shot == 7:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.3, 4
                elif args.eval_shot == 8:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.4, 4
                elif args.eval_shot == 9:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0, 0.4, 4
                elif args.eval_shot == 10:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.2, 4
            elif args.arch_for_all == 'wideres':
                if args.eval_shot == 1:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.5, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.2, 4
        elif args.data_for_all == 'cub':
            if args.arch_for_all == 'resnet18':
                if args.eval_shot == 1:
                    if args.eval_query_shots == 5:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.4, 4
                    elif args.eval_query_shots == 10:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.1, 4
                    elif args.eval_query_shots == 15:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.7, 0.7, 4
                    elif args.eval_query_shots == 20:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.7, 0.4, 4
                    elif args.eval_query_shots == 25:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.6, 0.6, 4
                    elif args.eval_query_shots == 30:
                        args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.6, 0.6, 4
                elif args.eval_shot == 2:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.3, 4
                elif args.eval_shot == 3:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.5, 0.1, 4
                elif args.eval_shot == 4:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.2, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.3, 4
                elif args.eval_shot == 6:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0, 4
                elif args.eval_shot == 7:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.1, 0.1, 4
                elif args.eval_shot == 8:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.4, 4
                elif args.eval_shot == 9:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.2, 4
                elif args.eval_shot == 10:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.1, 0.2, 4
            elif args.arch_for_all == 'wideres':
                if args.eval_shot == 1:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.6, 0.4, 4
                elif args.eval_shot == 5:
                    args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.2, 0.2, 4
    elif args.eval_n_ways in [10,20] and args.data_for_all == 'miniImagenet' and args.arch_for_all == 'resnet18':
        if args.eval_shot == 1:
            args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.4, 0.4, 4
        elif args.eval_shot == 5:
            args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau = 0.3, 0.3, 4

    print('%s  %s  TIM-GD  test_way %d  shot %d  enable_transduct %s  alpha1 %s  alpha2 %s  tau %s' %
          (args.data_for_all, args.arch_for_all, args.eval_n_ways, args.eval_shot, args.enable_transduct, args.transduct_alpha1, args.transduct_alpha2, args.transduct_tau))
    return args


def write_log(params, shot, l2n_mean, l2n_conf):
    with open(params.root_path + '/src/results_%d.txt' % (params.result_file_idx), 'a') as f:
        timestamp = time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime())
        exp_setting = '%s--%s--%s--%sway--%sshot--%s--a1 %s--a2 %s--tau %s--transduct %s' %(params.dataset_split_dir, params.eval_target_split_dir, params.model_arch,
                                                                      params.eval_n_ways, shot, params.eval_used_set,
                                                                      params.transduct_alpha1, params.transduct_alpha2, params.transduct_tau, params.enable_transduct)
        f.write( 'Time: %s | Setting: %s | Acc: %0.2f +- %0.2f \n' %(timestamp, exp_setting, l2n_mean*100, l2n_conf*100)  )


def set_args_shortcut(args):
    if args.data_for_all == 'miniImagenet':
        args.dataset_path = './data/mini_imagenet/'
        args.model_num_classes = 64
        args.dataset_split_dir = './split/mini/'
        if args.arch_for_all == 'resnet18':
            args.model_arch = 'resnet18'
            args.ckpt_path = './checkpoints/mini/softmax/resnet18'
            args.log = args.ckpt_path
        elif args.arch_for_all == 'wideres':
            args.model_arch = 'wideres'
            args.ckpt_path = './checkpoints/mini/softmax/wideres'
            args.log = args.ckpt_path
        else:
            raise ValueError('only resnet18 and wideres can be arch_for_all shortcut')
        args.pretrain = args.ckpt_path
    elif args.data_for_all == 'tiered_imagenet':
        args.dataset_path = './data/tiered_imagenet/data/'
        args.model_num_classes = 351
        args.dataset_split_dir = './split/tiered/'
        if args.arch_for_all == 'resnet18':
            args.model_arch = 'resnet18'
            args.ckpt_path = './checkpoints/tiered/softmax/resnet18'
            args.log = args.ckpt_path
        elif args.arch_for_all == 'wideres':
            args.model_arch = 'wideres'
            args.ckpt_path = './checkpoints/tiered/softmax/wideres'
            args.log = args.ckpt_path
        else:
            raise ValueError('only resnet18 and wideres can be arch_for_all shortcut')
        args.pretrain = args.ckpt_path
    elif args.data_for_all == 'cub':
        args.dataset_path = './data/cub/CUB_200_2011/images/'
        args.model_num_classes = 100
        args.dataset_split_dir = './split/cub/'
        if args.arch_for_all == 'resnet18':
            args.model_arch = 'resnet18'
            args.ckpt_path = './checkpoints/cub/softmax/resnet18'
            args.log = args.ckpt_path
        elif args.arch_for_all == 'wideres':
            args.model_arch = 'wideres'
            args.ckpt_path = './checkpoints/cub/softmax/wideres'
            args.log = args.ckpt_path
        else:
            raise ValueError('only resnet18 and wideres can be arch_for_all shortcut for cub')
        args.pretrain = args.ckpt_path
    else:
        raise ValueError('only miniImagenet, tiered_imagenet, cub can be data_for_all shortcut')

    return args


def get_one_hot(y_s):
    num_classes = torch.unique(y_s).size(0)
    eye = torch.eye(num_classes).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    return one_hot


def get_logs_path(model_path, method, shot):
    exp_path = '_'.join(model_path.split('/')[1:])
    file_path = os.path.join('tmp', exp_path, method)
    os.makedirs(file_path, exist_ok=True)
    return os.path.join(file_path, f'{shot}.txt')


def get_features(model, samples):
    features, _ = model(samples, True)
    features = F.normalize(features.view(features.size(0), -1), dim=1)
    return features


def get_loss(logits_s, logits_q, labels_s, lambdaa):
    Q = logits_q.softmax(2)
    y_s_one_hot = get_one_hot(labels_s)
    ce_sup = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1)  # Taking the mean over samples within a task, and summing over all samples
    ent_q = get_entropy(Q)
    cond_ent_q = get_cond_entropy(Q)
    loss = - (ent_q - cond_ent_q) + lambdaa * ce_sup
    return loss


def get_mi(probs):
    q_cond_ent = get_cond_entropy(probs)
    q_ent = get_entropy(probs)
    return q_ent - q_cond_ent


def get_entropy(probs):
    q_ent = - (probs.mean(1) * torch.log(probs.mean(1) + 1e-12)).sum(1, keepdim=True)
    return q_ent


def get_cond_entropy(probs):
    q_cond_ent = - (probs * torch.log(probs + 1e-12)).sum(2).mean(1, keepdim=True)
    return q_cond_ent


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    # handler = logging.StreamHandler()
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    file_handle_name = "file"
    # if file_handle_name in [h.name for h in logger.handlers]:
    #     return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def warp_tqdm(data_loader, disable_tqdm):
    if disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder='result/default'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')


def load_checkpoint(model, model_path, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path))
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(model_path))
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    state_dict = checkpoint['state_dict']
    names = []
    for k, v in state_dict.items():
        names.append(k)
    model.load_state_dict(state_dict)


def compute_confidence_interval(data, axis=0):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)
    pm = 1.96 * (std / np.sqrt(a.shape[axis]))
    return m, pm
