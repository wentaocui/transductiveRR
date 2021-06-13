import numpy as np
import torch
import torch.optim
import os

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet
from methods.transduct_protonet import transductProtoNet
from methods.transduct_matchingnet import transductMatchingNet
from methods.transduct_relationnet import transductRelationNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_baseline_file
import random

torch.manual_seed(10)
torch.cuda.manual_seed(10)
np.random.seed(10)
random.seed(10)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.adamlr)
        print('adam lr %s' % (params.adamlr))
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc, best_epoch = 0, start_epoch
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop(val_loader, epoch)
        if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
            print("refresh best model! save... \n")
            max_acc = acc
            best_epoch = epoch
            exp_setting = 'a1_%s_a2_%s_tao_%s_ssl_%s.tar' % (params.alpha1, params.alpha2, params.tau, params.ssl_weight)
            outfile = os.path.join(params.checkpoint_dir, exp_setting)
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch - best_epoch) > 30:
            print("no better model found for continuous 30 epochs, exit training! \n")
            return


    return model


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')
    os.environ['CUDA_VISIBLE_DEVICES'] = params.CUDA_VISIBLE_DEVICES

    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'
    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224
    optimization = 'Adam'

    if params.method in ['protonet', 'transductProtoNet', 'matchingnet', 'transductMatchingNet', 'relationnet',
                           'transductRelationNet', 'relationnet_softmax']:
        n_query = max(1, int(16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'transductProtoNet':
            model = transductProtoNet(model_dict[params.model], params.alpha1, params.alpha2, params.tau, params.ssl_weight, **train_few_shot_params)
        elif params.method == 'matchingnet':
            model = MatchingNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'transductMatchingNet':
            model = transductMatchingNet(model_dict[params.model], params.alpha1, params.alpha2, params.tau, params.ssl_weight, **train_few_shot_params)
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

            model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
        elif params.method == 'transductRelationNet':
            feature_model = lambda: model_dict[params.model](flatten=False)
            loss_type = 'mse'
            model = transductRelationNet(feature_model, params.alpha1, params.alpha2, params.tau, params.ssl_weight, loss_type=loss_type, **train_few_shot_params)

    elif params.method in ['maml', 'maml_approx']:
        n_query = max(1, int(16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True

        model = MAML(model_dict[params.model], approx=(params.method == 'maml_approx'), **train_few_shot_params)

    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    resume_file = get_baseline_file(params.checkpoint_dir)
    tmp = torch.load(resume_file)
    start_epoch = tmp['epoch'] + 1
    stop_epoch = start_epoch + 200
    model.load_state_dict(tmp['state'])

    model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)
