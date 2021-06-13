import torch
import numpy as np
import torch.optim
import torch.utils.data.sampler
import os
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.protonet import ProtoNet
from methods.transduct_protonet import transductProtoNet
from methods.transduct_matchingnet import transductMatchingNet
from methods.transduct_relationnet import transductRelationNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args
import random

torch.manual_seed(10)
torch.cuda.manual_seed(10)
np.random.seed(10)
random.seed(10)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15, adaptation=False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    if adaptation:
        scores = model.set_forward_adaptation(z_all, is_feature=True)
    else:
        scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc


if __name__ == '__main__':
    params = parse_args('test')
    os.environ['CUDA_VISIBLE_DEVICES'] = params.CUDA_VISIBLE_DEVICES

    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    if params.method == 'protonet':
        model = ProtoNet(model_dict[params.model], **few_shot_params)
    elif params.method == 'transductProtoNet':
        model = transductProtoNet(model_dict[params.model], params.alpha1, params.alpha2, params.tau, params.ssl_weight, **few_shot_params)
    elif params.method == 'matchingnet':
        model = MatchingNet(model_dict[params.model], **few_shot_params)
    elif params.method == 'transductMatchingNet':
        model = transductMatchingNet(model_dict[params.model], params.alpha1, params.alpha2, params.tau, params.ssl_weight, **few_shot_params)
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6':
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S':
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model](flatten=False)
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model = RelationNet(feature_model, loss_type=loss_type, **few_shot_params)
    elif params.method == 'transductRelationNet':
        feature_model = lambda: model_dict[params.model](flatten=False)
        loss_type = 'mse'
        model = transductRelationNet(feature_model, params.alpha1, params.alpha2, params.tau, params.ssl_weight, loss_type=loss_type, **few_shot_params)
    elif params.method in ['maml' , 'maml_approx', 'transductMaml']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True

        if 'Conv' in params.model:
            if params.dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84
        else:
            image_size = 224
        datamgr = SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)
        loadfile = configs.data_dir[params.dataset] + params.split + '.json'
        novel_loader = datamgr.get_data_loader(loadfile, aug=False)

        if params.method in ['maml' , 'maml_approx']:
            model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **few_shot_params )
        else:
            model = transductMAML(model_dict[params.model], approx=(params.method == 'maml_approx'), **few_shot_params, params=params)

    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    if params.transduct_mode == 'MT':
        modelfile = os.path.join(checkpoint_dir, 'best_model.tar')
    else:
        modelfile = os.path.join(checkpoint_dir, 'baseline_model.tar')
    tmp = torch.load(modelfile)
    model.load_state_dict(tmp['state'])

    if params.method in ['maml', 'maml_approx']:
        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        model.eval()
        acc_mean, acc_std = model.test_loop(novel_loader, 0, return_std=True)

        with open('./record/results.txt', 'a') as f:
            timestamp = time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime())
            aug_str = '-aug' if params.train_aug else ''
            aug_str += '-adapted' if params.adaptation else ''
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' % (
                params.dataset, params.split, params.model, params.method, aug_str, params.n_shot, params.train_n_way, params.test_n_way)
            acc_str = '%d Test Acc = %4.2f +- %4.2f' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
            f.write('Time: %s | Setting: %s | Acc: %s \n' % (timestamp, exp_setting, acc_str))
    else:
        split = params.split
        if params.save_iter != -1:
            split_str = split + "_" + str(params.save_iter)
        else:
            split_str = split

        novel_file = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                                  split_str + "_orig.hdf5")  # defaut split = novel, but you can also test base or val classes
        cl_data_file = feat_loader.init_loader(novel_file)

        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query=15, adaptation=params.adaptation, **few_shot_params)
            acc_all.append(acc)

            # if (i + 1) % 100 == 0:
            #     print('%d/%d Test Acc = %4.2f%% +- %4.2f%%' % (
            #     i + 1, iter_num, np.mean(np.asarray(acc_all)), 1.96 * np.std(np.asarray(acc_all)) / np.sqrt(iter_num)))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%% \n' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        with open('./record/results.txt', 'a') as f:
            timestamp = time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime())
            aug_str = '-aug' if params.train_aug else ''
            aug_str += '-adapted' if params.adaptation else ''
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test---a1_%s_a2_%s_tao_%s_ssl_%s' % (
            params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.train_n_way,
            params.test_n_way,
            params.alpha1, params.alpha2, params.tau, params.ssl_weight)
            acc_str = '%d Test Acc = %4.2f +- %4.2f' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
            f.write('Time: %s | Setting: %s | Acc: %s \n' % (timestamp, exp_setting, acc_str))
