import numpy as np
import torch
from torch.autograd import Variable
import os
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager
from io_utils import model_dict, parse_args
import random

torch.manual_seed(10)
torch.cuda.manual_seed(10)
np.random.seed(10)
random.seed(10)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


def save_features(model, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    for i, (x, y) in enumerate(data_loader):
        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()


if __name__ == '__main__':
    params = parse_args('save_features')
    os.environ['CUDA_VISIBLE_DEVICES'] = params.CUDA_VISIBLE_DEVICES

    image_size = 224
    split = params.split
    loadfile = configs.data_dir[params.dataset] + split + '.json'

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    if params.transduct_mode == 'MT':
        modelfile = os.path.join(checkpoint_dir, 'best_model.tar')
    else:
        modelfile = os.path.join(checkpoint_dir, 'baseline_model.tar')


    if params.save_iter != -1:
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                               split + "_" + str(params.save_iter) + ".hdf5")
    else:
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split + "_orig.hdf5")

    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False)

    if params.method in ['relationnet', 'relationnet_softmax', 'transductRelationNet']:
        if params.model == 'Conv4':
            model = backbone.Conv4NP()
        elif params.model == 'Conv6':
            model = backbone.Conv6NP()
        elif params.model == 'Conv4S':
            model = backbone.Conv4SNP()
        else:
            model = model_dict[params.model](flatten=False)
    else:
        model = model_dict[params.model]()

    model = model.cuda()
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.", "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.load_state_dict(state)
    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, data_loader, outfile)
