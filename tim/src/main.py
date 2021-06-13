import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from visdom_logger.logger import VisdomLogger
from utils import warp_tqdm, save_checkpoint
from trainer import Trainer
from eval import Evaluator
from optim import get_optimizer, get_scheduler
from models.ingredient import get_model
from utils import parse_args, set_args_shortcut

torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
random.seed(2020)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def main(params):
    os.chdir(params.root_path)
    cuda = params.cuda
    visdom_port = params.visdom_port

    os.environ['CUDA_VISIBLE_DEVICES'] = params.CUDA_VISIBLE_DEVICES
    device = torch.device("cuda" if cuda else "cpu")
    callback = None if visdom_port is None else VisdomLogger(port=visdom_port)
    torch.cuda.set_device(params.cuda_device)

    # create model
    print("=> Creating model '{}'".format(params.model_arch))
    model = torch.nn.DataParallel(get_model(params.model_arch, params.model_num_classes)).cuda()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = get_optimizer(model, params.optim_optimizer_name, params.optim_nesterov, params.optim_lr, params.optim_weight_decay)

    if params.pretrain:
        pretrain = os.path.join(params.pretrain, 'checkpoint.pth.tar')
        if os.path.isfile(pretrain):
            print("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            modelparams = checkpoint['state_dict']
            modelparams = {k: v for k, v in modelparams.items() if k in model_dict}
            model_dict.update(modelparams)
            model.load_state_dict(model_dict)
        else:
            print('[Warning]: Did not find pretrained model {}'.format(pretrain))

    if params.resume:
        resume_path = params.ckpt_path + '/checkpoint.pth.tar'
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print('[Warning]: Did not find checkpoint {}'.format(resume_path))
    else:
        start_epoch = 0
        best_prec1 = -1

    cudnn.benchmark = True

    # Data loading code
    evaluator = Evaluator(device=device, params=params)
    if params.evaluate:
        results = evaluator.run_full_evaluation(model, params.ckpt_path, params.eval_model_tag, params.eval_shot, params.eval_method, callback, params)
        return results

    # If this line is reached, then training the model
    trainer = Trainer(device, params.trainer_meta_val_iter, params.trainer_meta_val_way, params.trainer_meta_val_shot,
                 params.trainer_meta_val_query, params)
    scheduler = get_scheduler(optimizer=optimizer,
                              num_batches=len(trainer.train_loader),
                              epochs=params.epochs)
    tqdm_loop = warp_tqdm(list(range(start_epoch, params.epochs)),
                          disable_tqdm=params.disable_tqdm)
    for epoch in tqdm_loop:
        # Do one epoch
        trainer.do_epoch(model=model, optimizer=optimizer, epoch=epoch,
                         scheduler=scheduler, disable_tqdm=params.disable_tqdm,
                         callback=callback)

        # Evaluation on validation set
        if (epoch) % trainer.meta_val_interval == 0:
            prec1 = trainer.meta_val(model=model, disable_tqdm=params.disable_tqdm,
                                     epoch=epoch, callback=callback)
            print('Meta Val {}: {}'.format(epoch, prec1))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if not params.disable_tqdm:
                tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # Save checkpoint
        save_checkpoint(state={'epoch': epoch + 1,
                               'arch': params.model_arch,
                               'state_dict': model.state_dict(),
                               'best_prec1': best_prec1,
                               'optimizer': optimizer.state_dict()},
                        is_best=is_best,
                        folder=params.ckpt_path)
        if scheduler is not None:
            scheduler.step()

    # Final evaluation on test set
    results = evaluator.run_full_evaluation(model=model, model_path=params.ckpt_path)
    return results

if __name__=='__main__':
    params = parse_args()
    params = set_args_shortcut(params)
    accuracy_info = main(params)


