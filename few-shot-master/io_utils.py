import numpy as np
import os
import glob
import argparse
import backbone

model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101) 

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='miniImagenet',        help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model'       , default='ResNet18',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='transductProtoNet',   help='baseline/baseline++/protonet/transductProtoNet/matchingnet/transductMatchingNet/relationnet{_softmax}/transductRelationNet/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=1, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug'   , default=True,  type=bool, help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--alpha1', default=0, type=float, help='')
    parser.add_argument('--alpha2', default=0, type=float, help='')
    parser.add_argument('--tau', default=2, type=float, help='')
    parser.add_argument('--ssl_weight', default=1, type=float, help='')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0', type=str, help='')
    parser.add_argument('--transduct_mode', default='off', choices=['RR', 'MT', 'off'], help='')
    parser.add_argument('--auto_transduct', action='store_false', help='')

    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=10, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--resume_best', action='store_true', help='continue from previous best trained model')
        parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true') #never used in the paper
        parser.add_argument('--adamlr', default=5e-5, type=float, help='')
    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
    else:
       raise ValueError('Unknown script')

    params = parser.parse_args()

    if params.auto_transduct == True:
        params = load_transduct_params(params)

    return params

def load_transduct_params(params):
    if params.dataset == 'miniImagenet' and params.model == 'ResNet18' and params.train_n_way == 5 and params.test_n_way == 5:
        if params.method == 'transductProtoNet':
            if params.transduct_mode == 'RR':
                if params.n_shot == 1:
                    params.alpha1, params.alpha2, params.tau = 0.8, 0.8, 2
                elif params.n_shot == 5:
                    params.alpha1, params.alpha2, params.tau = 0.4, 0.4, 1
            elif params.transduct_mode == 'MT':
                if params.n_shot == 1:
                    params.alpha1, params.alpha2, params.tau = 0.5, 0.8, 0.8
                elif params.n_shot == 5:
                    params.alpha1, params.alpha2, params.tau = 0.4, 0.5, 0.5
            elif params.transduct_mode == 'off':
                params.alpha1, params.alpha2, params.tau = 0, 0, 2
        elif params.method == 'transductMatchingNet':
            if params.transduct_mode == 'RR':
                if params.n_shot == 1:
                    params.alpha1, params.alpha2, params.tau = 0.5, 0.5, 0.1
                elif params.n_shot == 5:
                    params.alpha1, params.alpha2, params.tau = 0.4, 0.6, 0.05
            elif params.transduct_mode == 'MT':
                if params.n_shot == 1:
                    params.alpha1, params.alpha2, params.tau = 0.5, 0.5, 0.1
                elif params.n_shot == 5:
                    params.alpha1, params.alpha2, params.tau = 0.4, 0.4, 0.05
            elif params.transduct_mode == 'off':
                params.alpha1, params.alpha2, params.tau = 0, 0, 0.1
        elif params.method == 'transductRelationNet':
            if params.transduct_mode == 'RR':
                if params.n_shot == 1:
                    params.alpha1, params.alpha2, params.tau = 0.5, 0.5, 5e-4
                elif params.n_shot == 5:
                    params.alpha1, params.alpha2, params.tau = 0.3, 0.4, 5e-4
            elif params.transduct_mode == 'MT':
                if params.n_shot == 1:
                    params.alpha1, params.alpha2, params.tau = 0.5, 0.7, 5e-4
                elif params.n_shot == 5:
                    params.alpha1, params.alpha2, params.tau = 0.5, 0.4, 3e-4
            elif params.transduct_mode == 'off':
                params.alpha1, params.alpha2, params.tau = 0, 0, 5e-4


    print('%s  %s  %s  test_way %d  shot %d  transduct_mode %s  alpha1 %s  alpha2 %s  tau %s' %
          (params.dataset, params.model, params.method, params.test_n_way, params.n_shot, params.transduct_mode, params.alpha1, params.alpha2, params.tau))
    return params


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_baseline_file(checkpoint_dir):
    return os.path.join(checkpoint_dir, 'baseline_model.tar')

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

