import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import math
import numpy as np
from config import config
import copy
from utils import check_cand

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for idx, primitive in enumerate(PRIMITIVES):
      op = OPS[primitive](C, stride, True)
      op.idx = idx
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=True))
      self._ops.append(op)

  def forward(self, x, rng):
    return self._ops[rng](x)


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)
    self._steps = steps
    self._multiplier = multiplier
    self._C = C
    self.out_C = self._multiplier * C
    self.reduction = reduction

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    self.time_stamp = 1 

    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, rngs):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, rngs[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
    return torch.cat(states[-self._multiplier:], dim=1)

class Network(nn.Module):
    def __init__(self, C=16, num_classes=10, layers=8, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C

        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input, rng):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, rng)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

if __name__ == '__main__':
    from  copy import deepcopy
    model = Network()
    operations = []
    for _ in range(config.edges):
        operations.append(list(range(config.op_num)))
    rng = [np.random.randint(len(config.blocks_keys)) for i in range(config.edges)]

    rngs = check_cand(rng, operations)
    x = torch.rand(4,3,32,32)
    logit = model(x, rngs)
    print('logit:{0}'.format(logit))