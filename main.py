"""
@author: Jiahua Wu
@contact: jhwu@shu.edu.cn
"""
import sys
import os

# from networkx import center
curPath = os.path.abspath(os.path.dirname('file'))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)
import random
import warnings
import argparse
import shutil
import os.path as osp
from typing import Tuple

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F

from codebase.modules import ImageClassifier, DomainDiscriminator,ConditionalDomainAdversarialLoss
from codebase.utils import experiment
from codebase.utils.data import ForeverDataIterator
from codebase.utils.metric import accuracy, ConfusionMatrix
from codebase.utils.meter import AverageMeter, ProgressMeter
from codebase.utils.logger import CompleteLogger
from codebase.utils.analysis import collect_feature, tsne, a_distance


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train for one epoch
def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
           model: ImageClassifier, domain_adv: ConditionalDomainAdversarialLoss, optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace
          ):
    losses = AverageMeter('Loss', ':3.2f')
    center_losses = AverageMeter('Center Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc',':3.1f')
    domain_accs = AverageMeter('Domain Acc',':3.1f')
    transfer_losses = AverageMeter('transfer_Loss', ':.2f')
    iters_loop = tqdm(range(args.iters_per_epoch), total=args.iters_per_epoch)

    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, cls_accs, domain_accs, center_losses, transfer_losses],
        prefix = "Epoch [{}/{}]".format(epoch,args.epochs),
        cur_tqdm = iters_loop
    )
    # switch to train mode
    model.train()
    domain_adv.train()

    for i in iters_loop:
        progress.display(i)
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, _, path_t = next(train_target_iter)[:3]
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        model.set_bn_domain(0)
        y_s, f_s = model(x_s)
        model.set_bn_domain(1)
        y_t, f_t = model(x_t)
        
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        
        loss = cls_loss + transfer_loss  
        cls_acc = accuracy(y_s, labels_s)[0]
        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))
        transfer_losses.update(transfer_loss.item(), x_s.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

def validate_dann(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_2 = AverageMeter('Loss2', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, losses_2, top2],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat_t = ConfusionMatrix(len(args.class_names))
        confmat_s = ConfusionMatrix(len(args.class_names))
    else:
        confmat_t = None
        confmat_s = None

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)
            model.set_bn_domain(1)
            output = model(images)
            loss = F.cross_entropy(output, target)
            acc1, = accuracy(output, target, topk=(1,))
            if confmat_t:
                confmat_t.update(target, output.argmax(1))
            model.set_bn_domain(0)
            output = model(images)
            loss_2 = F.cross_entropy(output, target)
            acc2, = accuracy(output, target, topk=(1,))
            # measure accuracy and record loss
            if confmat_s:
                confmat_s.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            losses_2.update(loss_2.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top2.update(acc2.item(), images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        print(' * Acc@2 {top2.avg:.3f}'.format(top2=top2))
        if confmat_s and confmat_t:
            print(confmat_t.format(args.class_names))
            print(confmat_s.format(args.class_names))

    return top1.avg, top2.avg
def main(args: argparse):
    log_file = osp.join(args.log,args.data+'_'+args.source[0]+'2'+args.target[0],str(args.seed)+"_"+args.experiment_name)
    logger = CompleteLogger(log_file, args.phase)
    args.device = device
    print(args)
    shutil.copy(os.path.abspath(sys.argv[0]) , log_file+'/')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting. '
                      'which can slow down your training considerably!'
                      'You may see unexpected behavior when restarting '
                      'from checkpoints. ')

    cudnn.benchmark = True
    #data
    train_transform = experiment.get_train_transform(args.train_resizing, scale =  args.scale, ratio = args.ratio,
                                                     random_horizontal_flip=not args.no_hflip,
                                                     random_color_jitter=False, resize_size=args.resize_size,
                                                     norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = experiment.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                                 norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)
    #
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        experiment.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    print(len(train_source_dataset), len(train_target_dataset), len(val_dataset), len(test_dataset))
    args.num_classes = num_classes
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = experiment.get_dsbn_model(args.arch, pretrain=not args.scratch, num_domains_bn= 2)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, num_domains_bn = 2, args = args,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    if args.randomized:
        domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(classifier.features_dim * args.num_classes, hidden_size=1024).to(device)
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier.features_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim, reduction='mean'
    ).to(device)
   
     # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1,_ = validate_dann(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0
    best_results = None
    for epoch in range(args.epochs):
        train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer, lr_scheduler, epoch, args)
        #evaluate on validation set
        acc1, acc2 = validate_dann(val_loader, classifier, args, device)
        # remember best acc@1 and checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1 or acc2 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1, acc2)

    print("best_acc1 = {:3.1f}, results = {}".format(best_acc1, best_results))

    # evaluate on test set
    checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1,acc2 = validate_dann(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))
    print("test_acc2 = {:3.1f}".format(acc2))

    logger.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=experiment.get_dataset_names(),
                        help='dataset: ' + ' | '.join(experiment.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=experiment.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(experiment.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    #
    parser.add_argument('--center_interita', type=float, default=0.7, help='centers inertia over batches')
    #
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=False, action='store_true', help='use entropy conditioning')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument("--experiment-name", type=str, default='1', help="Please name the experiment")
    args = parser.parse_args()
    main(args)
