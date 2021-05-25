import os
import sys
import time
import glob
import numpy as np
import torch
from utils import *
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from config import config
import shutil
import functools
print=functools.partial(print,flush=True)
from super_model import NetworkImageNet
import logging
import utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from megrec_imagenet import *
parser = argparse.ArgumentParser("Pytorch RLNAS ImageNet")
parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.25, help='init learning rate')
parser.add_argument('--min_lr', type=float, default=5e-4, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--classes', type=int, default=1000, help='number of classes')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--save', type=str, default='models', help='experiment name')
parser.add_argument('--data', metavar='DIR', default='./data/', help='path to dataset')
parser.add_argument('--workers', type=int, default=32, help='number of workers to load dataset')

args = parser.parse_args()

if args.local_rank == 0 and not os.path.exists(args.save):
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

time.sleep(1)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

IMAGENET_TRAINING_SET_SIZE = 1281167
train_iters = IMAGENET_TRAINING_SET_SIZE // args.batch_size

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    num_gpus = torch.cuda.device_count() 
    np.random.seed(args.seed)
    args.gpu = args.local_rank % num_gpus
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    group_name = 'darts_imagenet_supernet_training'
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    torch.distributed.init_process_group(backend='nccl', init_method='env://', group_name = group_name)
    args.world_size = torch.distributed.get_world_size()
    args.distributed = args.world_size > 1
    args.batch_size = args.batch_size // args.world_size
    criterion_smooth = utils.CrossEntropyLabelSmooth(args.classes, args.label_smooth).cuda()
    total_iters = args.epochs * train_iters

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
    train_transform = get_train_transform()
    train_dataset = get_imagenet_dataset(data_dir='/data/Dataset/ImageNet2012', train=True, transform=train_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers//args.world_size, pin_memory=True, sampler=train_sampler)

    operations = []
    for _ in range(config.edges):
        operations.append(list(range(config.op_num)))
    logging.info('operations={}'.format(operations))

    # Prepare model
    model, seed = NetworkImageNet(), args.seed
    model = model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    logging.info('arch = {}'.format(model.module.architecture()))
    optimizer, scheduler = utils.get_optimizer_schedule(model, args, total_iters)

    start_epoch = 0
    checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar, map_location={'cuda:0':'cuda:{}'.format(args.local_rank)})
        start_epoch = checkpoint['epoch'] + 1
        seed = checkpoint['seed']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        logging.info('{} load checkpoint..., epoch = {}, operations={}'.format(now, start_epoch, operations))

        # Reset the scheduler
        for _ in range(start_epoch):
            for _ in train_iters:
                if scheduler.get_lr()[0] > args.min_lr:
                    scheduler.step()

    # Save the base weights for computing angle
    if args.local_rank == 0:
        utils.save_checkpoint({'epoch':-1,
                         'state_dict': model.state_dict(),
                         'seed': seed
                         }, args.save)

    for epoch in range(start_epoch, args.epochs):
        # Supernet training
        seed = train(train_loader, optimizer, scheduler, model, criterion_smooth, operations, epoch, train_iters, seed, args)

        if args.local_rank==0 and (epoch+1)%5==0:
            utils.save_checkpoint( { 'epoch':epoch,
                                     'state_dict': model.state_dict(),
                                     'seed':seed}, args.save)


def train(train_loader, optimizer, scheduler, model, criterion, operations, epoch, train_iters, seed, args):
    objs, top1 = utils.AvgrageMeter(), utils.AvgrageMeter()
    model.train()
    for step, (image, target)in enumerate(train_loader):
        t0 = time.time()
        n = image.size(0)
        image = image.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        datatime = time.time() - t0

        # Uniform Sampling
        normal_cell, seed =  get_random_cand(seed, operations)
        redcution_cell, seed = get_random_cand(seed, operations)
        # Make sure each node has only two Predecessor nodes
        normal_cell = utils.check_cand(normal_cell, operations)
        redcution_cell = utils.check_cand(redcution_cell, operations)
        logits = model(image, normal_cell, redcution_cell)

        optimizer.zero_grad()
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0 and args.local_rank == 0:
            now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            logging.info('{} |=> Epoch={}, train: {} / {}, loss={:.2f}, acc={:.2f}, lr={}, datatime={:.2f}, seed={}' \
                  .format(now, epoch, step, train_iters, objs.avg, top1.avg, scheduler.get_lr()[0], float(datatime),
                          seed))

        if scheduler.get_last_lr()[0] > args.min_lr:
            scheduler.step()

    return seed

def get_random_cand(seed, operations):
    # Uniform Sampling
    cell = []
    for op in operations:
        np.random.seed(seed)
        k = np.random.randint(len(op))
        select_op = op[k]
        cell.append(select_op)
        seed += 1

    return cell, seed

if __name__ == '__main__':
  main() 