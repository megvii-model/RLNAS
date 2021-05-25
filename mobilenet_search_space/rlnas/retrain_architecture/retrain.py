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
import torch.utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from thop import profile
import pdb
import torch.distributed as dist
from tensorboardX import SummaryWriter
from model import Network
import numpy as np
import pickle
from config import config

IMAGENET_TRAINING_SET_SIZE = 1281167
IMAGENET_TEST_SET_SIZE = 50000

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=240, help='num of training epochs')
parser.add_argument('--total_iters', type=int, default=300000, help='total iters')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--model_id', type=str, default='', help='model_id')
parser.add_argument('--data', metavar='DIR', default='./data/', help='path to dataset')

args = parser.parse_args()

args.save = 'eval-{}'.format(args.save)
if args.local_rank == 0:
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

time.sleep(1)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(logdir=args.save)
CLASSES = 1000

IMAGENET_TRAINING_SET_SIZE = 1281167
train_iters = IMAGENET_TRAINING_SET_SIZE // args.batch_size
args.total_iters = train_iters * args.epochs

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

def get_arch_flops(op_flops_dict, cand):
    assert len(cand) == len(config.backbone_info) - 2
    preprocessing_flops = op_flops_dict['PreProcessing'][config.backbone_info[0]]
    postprocessing_flops = op_flops_dict['PostProcessing'][config.backbone_info[-1]]
    total_flops = preprocessing_flops + postprocessing_flops
    for i in range(len(cand)):
        inp, oup, img_h, img_w, stride = config.backbone_info[i+1]
        op_id = cand[i]
        if op_id >= 0:
          key = config.blocks_keys[op_id]
          total_flops += op_flops_dict[key][(inp, oup, img_h, img_w, stride)]
    return total_flops

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
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  torch.distributed.init_process_group(backend='nccl', init_method='env://')
  args.world_size = torch.distributed.get_world_size()
  args.distributed = args.world_size > 1
  args.batch_size = args.batch_size // args.world_size

  # The network architeture coding
  rngs = [int(id) for id in args.model_id.split(' ')]
  model = Network(rngs)
  op_flops_dict = pickle.load(open(config.flops_lookup_table, 'rb'))
  flops = get_arch_flops(op_flops_dict, rngs)
  params = utils.count_parameters_in_MB(model)
  model = model.cuda(args.gpu)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
  # model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

  arch = model.module.architecture()
  logging.info('rngs:{}, arch:{}'.format(rngs, arch))
  logging.info("flops = %fMB, param size = %fMB", flops/1e6, params)
  logging.info('batch_size:{}'.format(args.batch_size))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  all_parameters = model.parameters()
  weight_parameters = []
  for pname, p in model.named_parameters():
      if p.ndimension() == 4 or 'classifier.0.weight' in pname or 'classifier.0.bias' in pname:
          weight_parameters.append(p)
  weight_parameters_id = list(map(id, weight_parameters))
  other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

  optimizer = torch.optim.SGD(
    [{'params' : other_parameters},
    {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
    args.learning_rate,
    momentum=args.momentum,
    )
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.total_iters), last_epoch=-1)

  # Data loading code
  traindir = os.path.join(args.data, 'train')
  valdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
    ]))

  if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  else:
    train_sampler = None

  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers//args.world_size, pin_memory=True, sampler=train_sampler)

  val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)
  
  start_iter = 0
  best_acc_top1 = 0
  checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
  if os.path.exists(checkpoint_tar):
      logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
      checkpoint = torch.load(checkpoint_tar, map_location={'cuda:0':'cuda:{}'.format(args.local_rank)})
      start_epoch = checkpoint['epoch'] + 1
      best_acc_top1 = checkpoint['best_acc_top1']
      model.load_state_dict(checkpoint['state_dict'])
      logging.info("loaded checkpoint {} iters = {}" .format(checkpoint_tar, checkpoint['iters']))

  for iters in range(start_iter):
    scheduler.step()

  for epoch in range(start_epoch, args.epochs):
      train_acc, train_obj = train(train_loader, model, criterion_smooth, optimizer, scheduler, epoch)
      writer.add_scalar('Train/Loss', train_obj, epoch)
      writer.add_scalar('Train/LR', scheduler.get_lr()[0], epoch)
      if args.local_rank == 0:
        valid_acc_top1, valid_acc_top5, valid_obj = infer(val_loader, model.module, criterion, epoch)

        is_best = False
        if valid_acc_top1 > best_acc_top1:
          best_acc_top1 = valid_acc_top1
          is_best = True

        logging.info('valid_acc_top1: %f valid_acc_top5: %f best_acc_top1: %f', valid_acc_top1, valid_acc_top5, best_acc_top1)
        utils.save_checkpoint({
          'epoch': epoch,
          'state_dict': model.state_dict(),
          'best_acc_top1': best_acc_top1,
          'optimizer' : optimizer.state_dict(),
          }, args.save)

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for i, (image, target) in enumerate(train_loader):
    t0 = time.time()
    image = image.cuda(args.gpu, non_blocking=True)
    target = target.cuda(args.gpu, non_blocking=True)
    datatime = time.time() - t0

    logits = model(image)

    optimizer.zero_grad()
    loss = criterion(logits, target)
    loss_reduce = reduce_tensor(loss, 0, args.world_size)
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()
    scheduler.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = image.size(0)
    objs.update(loss_reduce.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)
    if i % args.report_freq == 0 and args.local_rank == 0:
      logging.info('Train epoch:%03d iters:%03d %e %f %f %f %f',epoch, i, objs.avg, top1.avg, top5.avg, scheduler.get_lr()[0], float(datatime))

  return top1.avg, objs.avg


def infer(val_loader, model, criterion, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for i, (image, target) in enumerate(val_loader):
      t0 = time.time()
      image = image.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)
      datatime = time.time() - t0

      logits = model(image)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = image.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if i % args.report_freq == 0:
        logging.info('Epoch: %03d valid %03d %e %f %f %f', epoch, i, objs.avg, top1.avg, top5.avg, float(datatime))
  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main() 
