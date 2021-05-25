import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import glob
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import joblib
import pdb
import pickle
from collections import defaultdict
from config import config
import copy

def broadcast(args, obj, src, group=torch.distributed.group.WORLD, async_op=False):
    print('local_rank:{}, obj:{}'.format(args.local_rank, obj))
    obj_tensor = torch.from_numpy(np.array(obj)).cuda()
    torch.distributed.broadcast(obj_tensor, src, group, async_op)
    obj = obj_tensor.cpu().numpy()
    print('local_rank:{}, tensor:{}'.format(args.local_rank, obj))
    return obj

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

def get_optimizer_schedule(model, args, total_iters):
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

    delta_iters = total_iters / (1.-args.min_lr / args.learning_rate)
    print('delta_iters={}'.format(delta_iters))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/delta_iters), last_epoch=-1)
    return optimizer, scheduler

def get_location(s, key):
    d = defaultdict(list)
    for k,va in [(v,i) for i,v in enumerate(s)]:
        d[k].append(va)
    return d[key]

def list_substract(list1, list2):
    list1 = [item for item in list1 if item not in set(list2)]
    return list1

def check_cand(cand, operations):
    cand = np.reshape(cand, [-1, config.edges])
    offset, cell_cand = 0, cand[0]
    for j in range(4):
        edges = cell_cand[offset:offset+j+2]
        edges_ops = operations[offset:offset+j+2]
        none_idxs = get_location(edges, 0)
        if len(none_idxs) < j:
            general_idxs = list_substract(range(j+2), none_idxs)
            num = min(j-len(none_idxs), len(general_idxs))
            general_idxs = np.random.choice(general_idxs, size=num, replace=False, p=None)
            for k in general_idxs:
                edges[k] = 0
        elif len(none_idxs) > j:
            none_idxs = np.random.choice(none_idxs, size=len(none_idxs)-j, replace=False, p=None)
            for k in none_idxs:
                if len(edges_ops[k]) > 1:
                    l = np.random.randint(len(edges_ops[k])-1)
                    edges[k] = edges_ops[k][l+1]
        offset += len(edges)

    return cell_cand.tolist()

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def save_checkpoint(state, save):
  if not os.path.exists(save):
    os.makedirs(save)
  filename = os.path.join(save, 'checkpoint_epoch_{}.pth.tar'.format(state['epoch']+1))
  torch.save(state, filename)
  print('Save CheckPoint....')


def save(model, save, suffix):
  torch.save(model.module.state_dict(), save)
  shutil.copyfile(save, 'weight_{}.pt'.format(suffix))

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  script_path = os.path.join(path, 'scripts')
  if scripts_to_save is not None and not os.path.exists(script_path):
    os.mkdir(script_path)
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def merge_ops(rngs):
    cand = []
    for rng in rngs:
        for r in rng:
          cand.append(r)
        cand += [-1]
    cand = cand[:-1]
    return cand

def split_ops(cand):
  cell, layer = 0, 0
  cand_ = [[]]
  for c in cand:
    if c == -1:
      cand_.append([])
      layer += 1
    else:
      cand_[layer].append(c)
  return cand_

def get_search_space_size(operations):
  comb_num = 1
  for j in range(len(operations)):
      comb_num *= len(operations[j])
  return comb_num

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

class Cifar10RandomLabels(dset.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.
  Params
  ------
  rand_seed: int
    Default 0. numpy random seed.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, rand_seed=0, num_classes=10, **kwargs):
    super(Cifar10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    self.rand_seed = rand_seed
    self.random_labels()

  def random_labels(self):
    labels = np.array(self.targets)
    print('num_classes:{}, random labels num:{}, random seed:{}'.format(self.n_classes, len(labels), self.rand_seed))
    np.random.seed(self.rand_seed)
    rnd_labels = np.random.randint(0, self.n_classes, len(labels))
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in rnd_labels]

    self.targets = labels