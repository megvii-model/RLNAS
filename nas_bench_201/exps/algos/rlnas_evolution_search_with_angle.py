##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
######################################################################################
# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019 #
######################################################################################
import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path

lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets import get_datasets, get_nas_search_loaders
from procedures import prepare_seed, prepare_logger, prepare_logger_test, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils import get_model_infos, obtain_accuracy
from log_utils import AverageMeter, time_string, convert_secs2time
from models import get_cell_based_tiny_net, get_search_spaces, get_sub_search_spaces
from nas_201_api import NASBench201API as API
import random
from weight_angle import get_arch_angle
from models.cell_searchs.genotypes import Structure

import os
import sys
import time
import numpy as np
import torch
import torch.utils

import sys

sys.setrecursionlimit(10000)
import argparse

import functools
import copy
import time
import functools
from collections import defaultdict

print = functools.partial(print, flush=True)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))

class EvolutionTrainer(object):
    def __init__(self,log_dir, logger, network, init_network, search_space, *,refresh=False):
        # evolution configuration
        self.select_num = 30
        self.op_num = 5
        self.edges = 6
        self.flops_limit = None
        self.population_num = 100
        self.mutation_num = 50
        self.crossover_num = 50
        self.m_prob = 0.1
        self.block_keys = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

        # model configuration
        self.network = network
        self.init_network = init_network
        self.search_space = search_space

        self.log_dir=log_dir
        self.logger = logger
        self.checkpoint_name=os.path.join(self.log_dir,'checkpoint.brainpkl')

        self.memory=[]
        self.candidates=[]
        self.vis_dict={}
        self.keep_top_k = {self.select_num:[],50:[]}
        self.epoch=0
        self.max_epochs = 20
        self.operations = [list(range(self.op_num)) for _ in range(self.edges)]

    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        info={}
        info['memory']=self.memory
        info['candidates']=self.candidates
        info['vis_dict']=self.vis_dict
        info['keep_top_k']=self.keep_top_k
        info['epoch']=self.epoch
        torch.save(info,self.checkpoint_name)
        self.logger.log('save checkpoint to {}'.format(self.checkpoint_name))

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info=torch.load(self.checkpoint_name)
        self.memory=info['memory']
        self.candidates = info['candidates']
        self.vis_dict=info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        self.logger.log('load checkpoint from {}'.format(self.checkpoint_name))
        return True

    def legal(self,cand):
        assert isinstance(cand,tuple) and len(cand)==self.edges
        if cand not in self.vis_dict:
            self.vis_dict[cand]={}
        info=self.vis_dict[cand]
        if 'visited' in info:
            return False

        if self.flops_limit is not None:
            pass

        self.vis_dict[cand]=info
        info['visited']=True

        return True

    def update_top_k(self,candidates,*,k,key,reverse=False):
        assert k in self.keep_top_k
        self.logger.log('select ......')
        t=self.keep_top_k[k]
        t+=candidates
        t.sort(key=key,reverse=reverse)
        self.keep_top_k[k]=t[:k]

    def stack_random_cand(self,random_func,*,batchsize=10):
        while True:
            cands=[random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand]={}
                else:
                    continue
                yield cand

    def stack_random_cand_crossover(self,random_func, max_iters, *,batchsize=10):
        cand_count = 0
        while True:
            if cand_count>max_iters:
                break
            cands=[random_func() for _ in range(batchsize)]
            cand_count += 1
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand]={}
                else:
                    continue
                info=self.vis_dict[cand]
            #for cand in cands:
                yield cand

    def random_can(self,num):
        self.logger.log('random select ........')
        candidates = []
        cand_iter=self.stack_random_cand(lambda:tuple(np.random.randint(self.op_num) for _ in range(self.edges)))
        while len(candidates)<num:
            cand=next(cand_iter)
            if not self.legal(cand):
                continue
            candidates.append(cand)
            self.logger.log('random {}/{}'.format(len(candidates),num))
        self.logger.log('random_num = {}'.format(len(candidates)))
        return candidates

    def get_mutation(self,k, mutation_num, m_prob):
        assert k in self.keep_top_k
        self.logger.log('mutation ......')
        res = []
        max_iters = mutation_num*10

        def random_func():
            cand=list(choice(self.keep_top_k[k]))
            for i in range(self.edges):
                if np.random.random_sample()<m_prob:
                    cand[i]=np.random.randint(1, self.op_num)
            return tuple(cand)

        cand_iter=self.stack_random_cand(random_func)
        while len(res)<mutation_num and max_iters>0:
            cand=next(cand_iter)
            if not self.legal(cand):
                continue
            res.append(cand)
            self.logger.log('mutation {}/{}'.format(len(res),mutation_num))
            max_iters-=1

        self.logger.log('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self,k, crossover_num):
        assert k in self.keep_top_k
        self.logger.log('crossover ......')
        res = []
        def random_func():
            p1=choice(self.keep_top_k[k])
            p2=choice(self.keep_top_k[k])
            return tuple(choice([i,j]) for i,j in zip(p1,p2))
        cand_iter=self.stack_random_cand_crossover(random_func, crossover_num)
        while len(res)<crossover_num:
            try:
                cand=next(cand_iter)
            except Exception as e:
                self.logger.log(e)
                break
            if not self.legal(cand):
                continue
            res.append(cand)
            self.logger.log('crossover {}/{}'.format(len(res),crossover_num))

        self.logger.log('crossover_num = {}'.format(len(res)))
        return res

    def transform_to_genotype(self, cand, search_space, max_nodes=4):
        genotypes = []

        edge_id = 0
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                op_id = cand[edge_id]
                op_name = search_space[op_id]
                xlist.append((op_name, j))
                edge_id += 1

            genotypes.append(tuple(xlist))

        return Structure(genotypes)

    def search(self):
        self.logger.log('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        if not self.load_checkpoint():
            self.candidates = self.random_can(self.population_num)
            self.save_checkpoint()

        while self.epoch<self.max_epochs:
            self.logger.log('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
                self.vis_dict[cand]['visited'] = True
                sampled_arch = self.transform_to_genotype(cand, self.search_space)
                self.vis_dict[cand]['angle'] = get_arch_angle(self.init_network, self.network, sampled_arch, self.search_space)
            self.update_top_k(self.candidates,k=self.select_num,key=lambda x:self.vis_dict[x]['angle'], reverse=True)
            self.update_top_k(self.candidates,k=50,key=lambda x:self.vis_dict[x]['angle'], reverse=True )

            self.logger.log('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[50])))
            for i,cand in enumerate(self.keep_top_k[50]):
                self.logger.log('No.{} {} Angle = {}'.format(i+1, cand, self.vis_dict[cand]['angle']))
                #ops = [config.blocks_keys[i] for i in cand]
                ops = [self.block_keys[i] for i in cand]
                self.logger.log(ops)

            mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num,self.crossover_num)
            rand = self.random_can(self.population_num - len(mutation) -len(crossover))
            self.candidates = mutation+crossover+rand

            self.epoch+=1
            self.save_checkpoint()

        self.logger.log(self.keep_top_k[self.select_num])
        self.logger.log('finish!')

        best_cand = self.keep_top_k[self.select_num][0]
        best_arch = self.transform_to_genotype(best_cand, self.search_space)
        angle = self.vis_dict[best_cand]['angle']

        return best_arch , angle

def update_init_model_weight(model, model_base_path, epoch, logger):
    checkpoint_path = model_base_path.format(epoch)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['search_model'])
    logger.log('Update initial model from:{}'.format(checkpoint_path))
    return model

def load_init_model(model, model_base_path, epoch, logger):
    checkpoint_path = model_base_path.format(epoch)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['search_model'])
    logger.log('load initial model from:{}'.format(checkpoint_path))
    return model

def get_best_arch_from_all_search_space(init_network, network, search_space, logger):
    with torch.no_grad():
        archs, angles = network.get_all_archs(), []
        random.shuffle(archs)
        for i, sampled_arch in enumerate(archs):
            # angle = get_arch_angle_exclude_none_path(init_network, network, sampled_arch, search_space)
            angle = get_arch_angle(init_network, network, sampled_arch, search_space)
            # logger.log('Step:{}, Genotype:{}, Angle:{}'.format(i, sampled_arch, angle))
            angles.append(angle)
        best_idx = np.argmax(angles)
        best_arch, max_angle = archs[best_idx], angles[best_idx]

        return best_arch, max_angle

def main(xargs):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    old_logger = prepare_logger(xargs)
    # args.save_dir = args.new_save_dir
    logger = prepare_logger_test(xargs)
    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1, xargs.rand_seed)
    config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
    search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset,
                                                            'configs/nas-benchmark/', \
                                                            (config.batch_size, config.test_batch_size), xargs.workers)
    logger.log(
        '||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset,
                                                                                                    len(search_loader),
                                                                                                    len(valid_loader),
                                                                                                    config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

    search_space = get_search_spaces('cell', xargs.search_space_name)
    model_config = dict2config({'name': 'SPOS', 'C': xargs.channel, 'N': xargs.num_cells,
                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                'space': search_space,
                                'affine': False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
    search_model = get_cell_based_tiny_net(model_config)

    w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
    a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999),
                                   weight_decay=xargs.arch_weight_decay)
    logger.log('w-optimizer : {:}'.format(w_optimizer))
    logger.log('a-optimizer : {:}'.format(a_optimizer))
    logger.log('w-scheduler : {:}'.format(w_scheduler))
    logger.log('criterion   : {:}'.format(criterion))
    flop, param = get_model_infos(search_model, xshape)
    # logger.log('{:}'.format(search_model))
    logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.log('search-space : {:}'.format(search_space))
    if xargs.arch_nas_dataset is None:
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log('{:} create API = {:} done'.format(time_string(), api))

    last_info, model_base_path, model_best_path = old_logger.path('info'), old_logger.path('model'), old_logger.path('best')
    model_base_path = str(model_base_path)+'_epoch_{}.pth'
    network, criterion = search_model.cuda(), criterion.cuda()
    init_network = deepcopy(network).cuda()
    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
        last_info = torch.load(str(last_info))
        start_epoch = last_info['epoch']
        checkpoint = torch.load(last_info['last_checkpoint'])
        network.load_state_dict(checkpoint['search_model'])
        init_network = load_init_model(init_network, model_base_path, 0, logger)
        w_scheduler.load_state_dict(checkpoint['w_scheduler'])
        w_optimizer.load_state_dict(checkpoint['w_optimizer'])
        a_optimizer.load_state_dict(checkpoint['a_optimizer'])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        exit(-1)

    # start evaluation
    start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup

    init_network = load_init_model(init_network, model_base_path, xargs.base_epoch, logger)
    evolution = EvolutionTrainer(xargs.new_save_dir, logger, network, init_network, search_space)
    genotype, angle = evolution.search()
    search_time.update(time.time() - start_time)

    logger.log('\n' + '-' * 100)
    # check the performance from the architecture dataset
    logger.log('SPOS : run {:} epochs, cost {:.1f} s, angle {},  last-geno is {:}.'.format(total_epoch, search_time.sum,
                                                                                           angle, genotype))
    if api is not None: logger.log('{:}'.format(api.query_by_arch(genotype, hp='200')))

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("SPOS")
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                        help='Choose between Cifar10/100 and ImageNet-16.')
    # channels and number-of-cells
    parser.add_argument('--search_space_name', type=str, help='The search space name.')
    parser.add_argument('--max_nodes', type=int, help='The maximum number of nodes.')
    parser.add_argument('--channel', type=int, help='The number of channels.')
    parser.add_argument('--num_cells', type=int, help='The number of cells in one stage.')
    parser.add_argument('--select_num', type=int, help='The number of selected architectures to evaluate.')
    parser.add_argument('--track_running_stats', type=int, choices=[0, 1],
                        help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--config_path', type=str, help='The path of the configuration.')
    # architecture leraning rate
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    # log
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir', type=str, help='Folder to save checkpoints and log.')
    parser.add_argument('--arch_nas_dataset', type=str,
                        help='The path to load the architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--print_freq', type=int, help='print frequency (default: 200)')
    parser.add_argument('--rand_seed', type=int, help='manual seed')
    parser.add_argument('--new_save_dir', type=str, help='Folder to save checkpoints and log.')
    parser.add_argument('--base_epoch', type=int, help='initial model')

    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
    main(args)