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

def broadcast(obj, src, group=torch.distributed.group.WORLD, async_op=False):
    obj_tensor = torch.from_numpy(np.array(obj)).cuda()
    torch.distributed.broadcast(obj_tensor, src, group, async_op)
    obj = obj_tensor.cpu().numpy()
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
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  print('Save CheckPoint....')

def save(model, model_path):
  torch.save(model.state_dict(), model_path)
  print('Save SuperNet....')

def load(model, model_path):
  model.load_state_dict(torch.load(model_path))
  print('Load SuperNet....')

def recalculate_bn(net, rngs, train_dataprovider, data_arr=None, data_for_bn='../train_dpflow_20000_images_for_BN.pkl', batchsize=64):
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)

    if not os.path.exists(data_for_bn):
      img_num = 0
      for step in range(1000):
          data_dict = next(train_dataprovider)
          image = data_dict['data']
          if data_arr is None:
              data_arr = image
          else:
              data_arr = np.concatenate((data_arr, image), 0)
          img_num = data_arr.shape[0]
          if img_num > 20000:
            break
      data_arr = data_arr[:20000, :, :, :]

      f = open(data_for_bn, 'wb')
      joblib.dump(data_arr, f)
      f.close()
    if data_arr is None:
        data_arr = joblib.load(open(data_for_bn, 'rb'))

    print('Compute BN, rng={}'.format(rngs))
    net.train()
    with torch.no_grad():
        for i in range(0, data_arr.shape[0], batchsize):
            data = data_arr[i:i+batchsize]
            data = torch.from_numpy(data).cuda()
            raw_logits = net(data, rngs)
            del raw_logits, data

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

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

def get_topk_str(rngs):
    cand = ''
    for r in rngs[0]:
      cand += str(r)
      cand += ' '
    cand = cand[:-1]
    return cand
