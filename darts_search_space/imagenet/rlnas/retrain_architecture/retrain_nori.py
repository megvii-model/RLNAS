import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import time
# import apex
import torch.distributed as dist

from torch.autograd import Variable
from model import NetworkImageNet as Network
from ntools.megtools.classification.config import DpflowProviderMaker, DataproProviderMaker
from tensorboardX import SummaryWriter
from thop import profile
import torchvision.datasets as datasets
from megrec_imagenet import *

parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
parser.add_argument('--workers', type=int, default=32, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='test', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--data', metavar='DIR', default='./data/', help='path to dataset')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--aug', type=str, default='NULL', help='lr scheduler, linear or cosine')

args, unparsed = parser.parse_known_args()

args.save = 'eval-{}'.format(args.save)

if args.local_rank == 0 and not os.path.exists(args.save):
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

time.sleep(1)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(logdir=args.save)

IMAGENET_TRAINING_SET_SIZE = 1281167
IMAGENET_TEST_SET_SIZE = 50000
CLASSES = 1000
train_iters = IMAGENET_TRAINING_SET_SIZE // args.batch_size
val_iters = IMAGENET_TEST_SET_SIZE // args.batch_size

# Average loss across processes for logging.
def reduce_tensor(tensor, device=0, world_size=1):
    tensor = tensor.clone()
    dist.reduce(tensor, device)
    tensor.div_(world_size)
    return tensor


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    args.gpu = args.local_rank % num_gpus
    torch.cuda.set_device(args.gpu)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.distributed = args.world_size > 1
    args.batch_size = args.batch_size // args.world_size
    # Prepare data
    # traindir = os.path.join(args.data, 'train')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # train_dataset = utils.ImageNetWithRandomLabels(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    train_transform = get_train_transform(aug=args.aug)
    train_dataset = get_imagenet_dataset(data_dir='/data/Dataset/ImageNet2012', train=True, transform=train_transform)
    eval_transform = get_eval_transform(aug=args.aug)
    val_dataset = get_imagenet_dataset(data_dir='/data/Dataset/ImageNet2012', train=False, transform=eval_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers//args.world_size, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers//args.world_size, pin_memory=True)

    genotype = eval("genotypes.%s" % args.arch)
    logging.info('---------Genotype---------')
    logging.info(genotype)
    logging.info('--------------------------')
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda(args.gpu)
    # model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, broadcast_buffers=False)
    model_profile = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model_profile = model_profile.cuda(args.gpu)
    model_input_size_imagenet = (1, 3, 224, 224)
    model_profile.drop_path_prob = 0
    flops, _ = profile(model_profile, model_input_size_imagenet)
    logging.info("flops = %fMB, param size = %fMB", flops, utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    start_epoch = 0
    best_acc_top1 = 0
    best_acc_top5 = 0
    checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
        logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar, map_location={'cuda:0': 'cuda:{}'.format(args.local_rank)})
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint {} epoch = {}".format(checkpoint_tar, checkpoint['epoch']))

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            logging.info('Wrong lr type, exit')
            sys.exit(1)

        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
        model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()
        train_acc, train_obj = train(train_loader, model, criterion_smooth, optimizer)

        writer.add_scalar('Train/Loss', train_obj, epoch)
        writer.add_scalar('Train/LR', current_lr, epoch)

        if args.local_rank == 0:
            valid_acc_top1, valid_acc_top5, valid_obj = infer(val_loader, model.module, criterion)
            is_best = False
            if valid_acc_top5 > best_acc_top5:
                best_acc_top5 = valid_acc_top5
            if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
                is_best = True

            logging.info('Valid_acc_top1: %f', valid_acc_top1)
            logging.info('Valid_acc_top5: %f', valid_acc_top5)
            logging.info('best_acc_top1: %f', best_acc_top1)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds.', epoch_duration)

            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save)


def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs - epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_loader, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()

    for i, (image, target) in enumerate(train_loader):
        t0 = time.time()
        image = image.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        datatime = time.time() - t0

        b_start = time.time()
        logits, logits_aux = model(image)
        optimizer.zero_grad()
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss_reduce = reduce_tensor(loss, 0, args.world_size)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = image.size(0)
        objs.update(loss_reduce.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if i % args.report_freq == 0 and args.local_rank == 0:
            logging.info('TRAIN Step: %03d/%03d Objs: %e R1: %f R5: %f BTime: %.3fs Datatime: %.3f',
                         i, train_iters, objs.avg, top1.avg, top5.avg, batch_time.avg, float(datatime))

    return top1.avg, objs.avg


def infer(val_loader, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        t0 = time.time()
        image = image.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        datatime = time.time() - t0

        with torch.no_grad():
            logits, _ = model(image)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = image.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if i % args.report_freq == 0:
            logging.info('VALID Step: %03d/%03d Objs: %e R1: %f R5: %f Datatime: %.3f', i, val_iters*args.world_size, objs.avg,
                         top1.avg, top5.avg, float(datatime))

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
