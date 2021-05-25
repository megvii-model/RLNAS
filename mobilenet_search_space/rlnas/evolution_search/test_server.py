#!/usr/bin/env python3
from mq_server_base import MessageQueueServerBase, MessageQueueClientBase
import argparse
import os
import time
import re
import gc

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from config import config
from super_model import SuperNetwork
import numpy as np
import utils
from utils import *
import functools
import sys

print = functools.partial(print, flush=True)
import pdb
import logging

class TorchMonitor(object):
    def __init__(self):
        self.obj_set = set()
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj not in self.obj_set:
                self.obj_set.add(obj)

    def find_leak_tensor(self):
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj not in self.obj_set:
                print(obj.size())


class TestClient(MessageQueueClientBase):
    def __init__(self, *, random):
        if random:
            super().__init__(config.host, config.port, config.username,
                             config.random_test_send_pipe, config.random_test_recv_pipe)
        else:
            super().__init__(config.host, config.port, config.username,
                             config.test_send_pipe, config.test_recv_pipe)

    def send(self, cand):
        assert isinstance(cand, tuple)
        return super().send(cand)


class TestServer(MessageQueueServerBase):
    def __init__(self, batchsize, *, random):
        if random:
            super().__init__(config.host, config.port, config.username,
                             config.random_test_send_pipe, config.random_test_recv_pipe)
        else:
            super().__init__(config.host, config.port, config.username,
                             config.test_send_pipe, config.test_recv_pipe)

        self._recompile_net()

    def _recompile_net(self):

        assert (os.path.exists(config.net_cache))
        self.model = torch.nn.DataParallel(SuperNetwork()).cuda()
        checkpoint = torch.load(config.net_cache)
        self.model.load_state_dict(checkpoint['state_dict'])
        logging.info('loading model from path:{0}'.format(config.net_cache))
        assert (os.path.exists(config.initial_net_cache))
        self.initial_model = torch.nn.DataParallel(SuperNetwork()).cuda()
        init_checkpoint = torch.load(config.initial_net_cache)
        self.initial_model.load_state_dict(init_checkpoint['state_dict'])
        logging.info('loading initial model from path:{0}'.format(config.initial_net_cache))

    # Algorithm 1
    def get_arch_vector(self, model, cand):
        conv_bn = torch.reshape(model.conv_bn[0].weight.data, (-1,))
        conv1 = torch.reshape(model.MBConv_ratio_1.conv[0].weight.data, (-1,))
        conv2 = torch.reshape(model.MBConv_ratio_1.conv[3].weight.data, (-1,))
        conv_1x1_bn = torch.reshape(model.conv_1x1_bn[0].weight.data, (-1,))
        classifier = torch.reshape(model.classifier[0].weight.data, (-1,))
        arch_vector = [conv_bn, conv1, conv2, conv_1x1_bn, classifier]
        # block-like weight vector construction procedure is adopted
        for i, c in enumerate(cand):
            if c >= 0:
                conv1 = torch.reshape(model.features[i]._ops[c].conv[0].weight.data, (-1,))
                conv1_bn_weight = torch.reshape(model.features[i]._ops[c].conv[1].weight.data, (-1,))
                conv1_bn_bias = torch.reshape(model.features[i]._ops[c].conv[1].bias.data, (-1,))

                conv2 = torch.reshape(model.features[i]._ops[c].conv[3].weight.data, (-1,))
                conv2_bn_weight = torch.reshape(model.features[i]._ops[c].conv[4].weight.data, (-1,))
                conv2_bn_bias = torch.reshape(model.features[i]._ops[c].conv[4].bias.data, (-1,))

                conv3 = torch.reshape(model.features[i]._ops[c].conv[6].weight.data, (-1,))
                arch_vector += [torch.cat([conv1, conv1_bn_weight, conv1_bn_bias, conv2, conv2_bn_weight, conv2_bn_bias, conv3], dim=0)]
            elif c == -1:
                input_channel = model.features[i].input_channel
                ouput_channel = model.features[i].output_channel
                identity_param = torch.eye(ouput_channel, input_channel)
                arch_vector += [identity_param.reshape(-1).cuda()]
            else:
                raise Exception("Invalid operators !")

        arch_vector = torch.cat(arch_vector, dim=0)
        return arch_vector

    def get_angle(self, initial_model, model, cand):
        cosine = nn.CosineSimilarity(dim=0).cuda()
        vec1 = self.get_arch_vector(initial_model.module, cand)
        vec2 = self.get_arch_vector(model.module, cand)
        angle = torch.acos(cosine(vec1, vec2))
        return angle.cpu().item()

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
            os._exit(1)
            res['status'] = 'failure'
            return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=200)
    parser.add_argument('-p', '--process', type=int, default=1)
    parser.add_argument('-r', '--reset', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--log_path', type=str, default='./log', help='log path')

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.log_path, 'search_log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    train_server = TestServer(args.batch_size, random=args.random)
    train_server.run(args.process, reset_pipe=args.reset)


if __name__ == "__main__":
    try:
        main()
    except:
        import traceback

        traceback.print_exc()
        print(flush=True)
        os._exit(1)
