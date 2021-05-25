#!/usr/bin/env python3
from mq_server_base import MessageQueueServerBase,MessageQueueClientBase
from multiprocessing import Process
from multiprocessing import Queue
from ntools.utils.misc import natural_sort
import argparse
import pickle
import shutil

import os
import time
import hashlib
import glob
import re
import gc

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from config import config
from super_model import NetworkImageNet

from copy import deepcopy
from operations import *
import utils
import logging
import sys
import functools
print=functools.partial(print,flush=True)

class TorchMonitor(object):
    def __init__(self):
        self.obj_set=set()
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj not in self.obj_set:
                self.obj_set.add(obj)
    def find_leak_tensor(self):
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj not in self.obj_set:
                print(obj.size())

class TestClient(MessageQueueClientBase):
    def __init__(self):
        super().__init__(config.host, config.port, config.username,
                            config.test_send_pipe, config.test_recv_pipe)
    def send(self,cand):
        assert isinstance(cand,tuple)
        return super().send(cand)

class TestServer(MessageQueueServerBase):
    def __init__(self, batchsize):
        super().__init__(config.host, config.port, config.username,
                            config.test_send_pipe, config.test_recv_pipe)
        self._recompile_net()
        self.batchsize=batchsize

    def _recompile_net(self):

        model = NetworkImageNet()
        initial_model = deepcopy(model)
        model = torch.nn.DataParallel(model)
        initial_model = torch.nn.DataParallel(initial_model)
        self.device = 'cpu'
        if torch.cuda.is_available():
            model = model.cuda()
            initial_model = initial_model.cuda()
            self.device = 'cuda'
        # model = nn.DataParallel(model)
        # initial_model = nn.DataParallel(initial_model)
        assert os.path.exists(config.net_cache)
        logging.info('loading model {} ..........'.format(config.net_cache))
        checkpoint = torch.load(config.net_cache,map_location='cpu')
        logging.info('loading states....')
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint {} epoch = {}" .format(config.net_cache, checkpoint['epoch']))

        assert os.path.exists(config.initial_net_cache)
        logging.info('loading initial model {} ..........'.format(config.initial_net_cache))
        checkpoint = torch.load(config.initial_net_cache, map_location='cpu')
        logging.info('loading states....')
        initial_model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint {} epoch = {}".format(config.initial_net_cache, checkpoint['epoch']))

        self.model = model.module
        self.initial_model = initial_model.module

    def eval(self, cand):
        cand = list(cand)
        logging.info('cand={}'.format(cand))
        res = self._test_candidate(cand)
        return res

    def _test_candidate(self, cand):
        res = dict()
        try:
            t0 = time.time()
            angle = self.get_angle(self.initial_model, self.model, cand)
            logging.info('cand={}, angle={}'.format(cand, angle))
            logging.info('time: {}s'.format(time.time() - t0))
            res = {'status': 'success', 'angle': angle}
            return res
        except:
            import traceback
            traceback.print_exc()
            res['status'] = 'failure'
            return res

    def get_arch_vector(self, model, normal_cand, reduction_cand):
        cand = []
        for layer in range(config.layers):
            if layer in [config.layers//3, 2*config.layers//3]:
                cand.append(deepcopy(reduction_cand))
            else:
                cand.append(deepcopy(normal_cand))

        arch_vector, extra_params = [], []
        # Collect extra parameters
        # stem0
        stem0 = torch.cat([model.stem0[0].weight.data.reshape(-1), model.stem0[1].weight.data.reshape(-1), model.stem0[1].bias.data.reshape(-1),\
                           model.stem0[3].weight.data.reshape(-1), model.stem0[4].weight.data.reshape(-1), model.stem0[4].bias.data.reshape(-1)])
        extra_params += [stem0]
        stem1 = torch.cat([model.stem1[1].weight.data.reshape(-1), model.stem1[2].weight.data.reshape(-1), model.stem1[2].bias.data.reshape(-1)])
        extra_params += [stem1]
        for i in range(len(model.cells)):
            # Collect extra parameters
            if isinstance(model.cells[i].preprocess0, FactorizedReduce):
                s0 = torch.cat([model.cells[i].preprocess0.conv_1.weight.data.reshape(-1),
                                model.cells[i].preprocess0.conv_2.weight.data.reshape(-1),
                                model.cells[i].preprocess0.bn.weight.data.reshape(-1),
                                model.cells[i].preprocess0.bn.bias.data.reshape(-1)])
            else:
                s0 = torch.cat([model.cells[i].preprocess0.op[1].weight.data.reshape(-1),
                                model.cells[i].preprocess0.op[2].weight.data.reshape(-1),
                                model.cells[i].preprocess0.op[2].bias.data.reshape(-1)])
            s1 = torch.cat([model.cells[i].preprocess1.op[1].weight.data.reshape(-1),
                            model.cells[i].preprocess1.op[2].weight.data.reshape(-1),
                            model.cells[i].preprocess1.op[2].bias.data.reshape(-1)])
            extra_params += [s0, s1]

            # Collect weight vecors of all paths
            param_list = []
            for path in config.paths:
                param_cell = []
                for index in range(1, len(path)):
                    j = path[index]
                    k = path[index - 1]
                    assert (j >= 2)
                    offset = 0
                    for tmp in range(2, j):
                        offset += tmp

                    if cand[i][k + offset] == config.NONE:  # None
                        param_cell = []
                        break

                    elif cand[i][k + offset] == config.MAX_POOLING_3x3 or cand[i][
                        k + offset] == config.AVG_POOL_3x3:  # pooling
                        shape = model.cells[i]._ops[k + offset]._ops[4].op[1].weight.data.shape
                        shape = [shape[0], shape[2], shape[3]]
                        pooling_param = torch.ones(shape) * (1 / 9.)
                        param_cell += [torch.cat([pooling_param.reshape(-1).to(self.device),
                                                 model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]][1].weight.data.reshape(-1),
                                                 model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]][1].bias.data.reshape(-1)])]
                    elif cand[i][k + offset] == config.SKIP_CONNECT:  # identity
                        # pass
                        if isinstance(model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]], FactorizedReduce):
                             param_cell +=  [torch.cat([model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].conv_1.weight.data.reshape(-1),
                                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].conv_2.weight.data.reshape(-1),
                                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].bn.weight.data.reshape(-1),
                                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].bn.bias.data.reshape(-1)])]
                        elif isinstance(model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]], Identity):
                            shape = model.cells[i]._ops[k + offset]._ops[4].op[6].weight.data.shape
                            identity_param = torch.eye(shape[0], shape[0])
                            param_cell += [identity_param.reshape(-1).to(self.device)]
                        else:
                            raise Exception("Invalid operators !")

                    elif cand[i][k + offset] == config.SEP_CONV_3x3 or cand[i][
                        k + offset] == config.SEP_CONV_5x5:  # sep conv
                        conv1 = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[1].weight.data, (-1,))
                        conv2 = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[2].weight.data, (-1,))
                        conv3 = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[5].weight.data, (-1,))
                        conv4 = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[6].weight.data, (-1,))
                        conv_cat = torch.cat([conv1, conv2, conv3, conv4])

                        bn1_weight = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[3].weight.data, (-1,))
                        bn1_bias = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[3].bias.data, (-1,))

                        bn2_weight = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[7].weight.data, (-1,))
                        bn2_bias = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[7].bias.data, (-1,))

                        bn_cat = torch.cat([bn1_weight, bn1_bias, bn2_weight, bn2_bias])

                        param_cell += [conv_cat]
                        param_cell += [bn_cat]

                    elif cand[i][k + offset] == config.DIL_CONV_3x3 or cand[i][k + offset] == config.DIL_CONV_5x5:
                        conv1 = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[1].weight.data, (-1,))
                        conv2 = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[2].weight.data, (-1,))
                        conv_cat = torch.cat([conv1, conv2])

                        bn1_weight = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[3].weight.data, (-1,))
                        bn1_bias = torch.reshape(
                            model.cells[i]._ops[k + offset]._ops[cand[i][k + offset]].op[3].weight.data, (-1,))

                        bn_cat = torch.cat([bn1_weight, bn1_bias])

                        param_cell += [conv_cat]
                        param_cell += [bn_cat]
                    else:
                        raise Exception("Invalid operators !")

                # Get weight vector of a path
                if len(param_cell) != 0:
                    param_list.append(torch.cat(param_cell))

            # Get weight vector of a cell
            if len(param_list) != 0:
                arch_vector.append(torch.cat(param_list, dim=0))

        # Collect extra parameters
        extra_params.append(torch.cat([model.classifier.weight.data.reshape(-1),
                                       model.classifier.bias.data.reshape(-1)]))
        arch_vector += extra_params

        # Get weight vector of the whole model
        if len(arch_vector) != 0:
            arch_vector = torch.cat(arch_vector, dim=0)
        return arch_vector

    def get_angle(self, initial_model, model, cand):
        cosine = torch.nn.CosineSimilarity(dim=0).to(self.device)
        normal_cell = cand[:config.edges]
        redcution_cell = cand[config.edges:]
        vec1 = self.get_arch_vector(initial_model, normal_cell, redcution_cell)
        vec2 = self.get_arch_vector(model, normal_cell, redcution_cell)
        return torch.acos(cosine(vec1, vec2)).cpu().item()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-p', '--process', type=int, default=1)
    parser.add_argument('-r', '--reset', action='store_true')
    parser.add_argument('--save', type=str, default='log', help='experiment name')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')

    args=parser.parse_args()

    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'test_server_log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    train_server = TestServer(args.batch_size)
    train_server.run(args.process, reset_pipe=args.reset)

if __name__ == "__main__":
    try:
        main()
    except:
        import traceback
        traceback.print_exc()
        print(flush=True)
        os._exit(1)



