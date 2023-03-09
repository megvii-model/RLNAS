##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
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
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, prepare_logger_test, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from nas_201_api  import NASBench201API as API
import pdb
import time
import scipy
import scipy.stats
from weight_angle import get_arch_angle
from itertools import product
from models.cell_searchs.genotypes import Structure
import torchvision.datasets as dset
import torchvision.transforms as transforms
import pickle
import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load(checkpoint_path, model):
  checkpoint  = torch.load(checkpoint_path)
  model.load_state_dict( checkpoint['search_model'] )

# ground truth acc
def get_gt_accs(network, sample_archs, api):
    # all_archs = network.get_all_archs()
    acc_dict = {}
    for i, genotype in enumerate(sample_archs):
        acc = get_arch_real_acc(api, genotype)
        acc_dict[genotype.tostr()] = acc

    return acc_dict

def get_arch_real_acc(api, genotype, dataset='cifar10', acc_type='ori-test'):
    id = api.query_index_by_arch(genotype)
    info = api.query_by_index(id, hp='200')
    accs = []
    for (dataset, acc_type) in (('cifar10', 'ori-test'), ('cifar100', 'x-test'), ('ImageNet16-120', 'x-test')):
        acc = info.get_metrics(dataset, acc_type)['accuracy']
        accs.append(acc)

    return accs

def get_angles(logger, init_model, model, sample_archs, search_space, api, dataset):
    # all_archs = model.get_all_archs()
    arch_angles = {}
    gt_accs = {}
    if dataset == 'cifar10':
        acc_type = 'ori-test'
    elif dataset == 'cifar100':
        acc_type = 'x-test'
    elif dataset == 'ImageNet16-120':
        acc_type = 'x-test'
    else:
        raise NotImplementedError

    for i, genotype in enumerate(sample_archs):
        angle = get_arch_angle(init_model, model, genotype, search_space)
        acc = get_arch_real_acc(api, genotype, dataset, acc_type)
        logger.log('[{:}] cal angle : angle={}, acc: {}, | {:}'.format(i, angle, acc, genotype))
        arch_angles[genotype.tostr()] = angle
        gt_accs[genotype.tostr()] = acc

    return arch_angles, gt_accs


def cal_kendall_tau(gt_accs_dict, metric_dict):
    # delete key whose score value is np.nan
    keys = list(metric_dict.keys())
    for key in keys:
        if np.isnan(metric_dict[key]):
            gt_accs_dict.pop(key)
            metric_dict.pop(key)

    gt_accs_dict = sorted(gt_accs_dict.items(), key=lambda d: d[1], reverse=True)
    metric_dict = sorted(metric_dict.items(), key=lambda d: d[1], reverse=True)

    rank = 1
    metric_rank = {}
    for value in metric_dict:
        metric_rank[value[0]] = rank
        rank += 1

    rank = 1
    gt_accs_rank_list = []
    metric_rank_list = []
    for value in gt_accs_dict:
        gt_accs_rank_list.append(rank)
        metric_rank_list.append(metric_rank[value[0]])
        rank += 1

    metric_tau = scipy.stats.kendalltau(gt_accs_rank_list, metric_rank_list)[0]

    return metric_tau

def transform_to_genotype(candidate, search_space, max_nodes=4):
    genotypes = []

    edge_id = 0
    for i in range(1, max_nodes):
        xlist = []
        for j in range(i):
            op_id = candidate[edge_id]
            op_name = search_space[op_id]
            xlist.append((op_name, j))
            edge_id += 1

    genotypes.append(tuple(xlist))

    return Structure(genotypes)


def get_all_archs(operations, search_space, max_nodes=4):
    all_archs = []
    for arch in product(*operations):
        arch_genotype = transform_to_genotype(arch, search_space, max_nodes)
        all_archs.append(arch_genotype)

    return all_archs

def load_init_model(model, model_base_path, epoch, logger):
    checkpoint_path = model_base_path.format(epoch)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['search_model'])
    logger.log('load initial model from:{}'.format(checkpoint_path))
    return model

def main(xargs):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads( xargs.workers )
    prepare_seed(xargs.rand_seed)
    old_logger = prepare_logger(xargs)
    # args.save_dir = args.new_save_dir
    logger = prepare_logger_test(xargs)
    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1, xargs.rand_seed)
    config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
    search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', \
                                        (config.batch_size, config.test_batch_size), xargs.workers)

    search_space = get_search_spaces('cell', xargs.search_space_name)
    model_config = dict2config({'name': 'SPOS', 'C': xargs.channel, 'N': xargs.num_cells,
                              'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                              'space'    : search_space,
                              'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)

    model = get_cell_based_tiny_net(model_config)
    w_optimizer, w_scheduler, criterion = get_optim_scheduler(model.get_weights(), config)

    if xargs.arch_nas_dataset is None:
        api = None
    else:
        api = API(xargs.arch_nas_dataset)

    last_info, model_base_path, model_best_path = old_logger.path('info'), old_logger.path('model'), old_logger.path('best')
    model_base_path = str(model_base_path)+'_epoch_{}.pth'
    network = model.cuda()
    init_network = deepcopy(network).cuda()
    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
        last_info = torch.load(str(last_info))
        start_epoch = last_info['epoch']
        checkpoint = torch.load(last_info['last_checkpoint'])
        network.load_state_dict(checkpoint['search_model'])
        init_network = load_init_model(init_network, model_base_path, xargs.base_epoch, logger)
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        exit(-1)

    all_archs = model.get_all_archs()
    angles_dict, all_gt_accs_dict = get_angles(logger, init_network, network, all_archs, search_space, api, xargs.dataset)

    all_datasets = ("cifar10", 'cifar100', 'ImageNet16-120')
    for idataset, dataset_name in enumerate(all_datasets):
        gt_accs_dict = {k: v[idataset] for k, v in all_gt_accs_dict.items()}
        angles_tau = cal_kendall_tau(gt_accs_dict, angles_dict)
        logger.log(f'angles tau for {dataset_name}:{angles_tau}')

        df = pandas.DataFrame({"angle": angles_dict, "acc":gt_accs_dict})
        top_angle = df["angle"].max()
        top_angle_acc = df[df["angle"] == top_angle]["acc"]
        fig = plt.figure(figsize=(8,3))
        fig.suptitle(f"Angle v.s. Accuracy on {dataset_name}, tau = {angles_tau:.3f}", fontsize=16)
        axs = []
        for i, xlims in enumerate(((0.7, 1.6), (1.46, 1.54), (1.5, 1.53))):
            ax = plt.subplot(1, 3, i + 1)
            axs.append(ax)
            plt.scatter(df["angle"], df["acc"], marker=".")
            plt.scatter(top_angle, top_angle_acc)
            plt.xlim(*xlims)
            offset = 2 - i * 0.5
            ylims = (df[df["angle"] > xlims[0]]["acc"].min() - offset, df[df["angle"] < xlims[1]]["acc"].max() + offset)
            plt.ylim(*ylims)

        for i in range(2):
            axs[i].add_patch(Rectangle(
                (axs[i+1].get_xlim()[0], axs[i+1].get_ylim()[0]),
                axs[i+1].get_xlim()[1] - axs[i+1].get_xlim()[0],
                axs[i+1].get_ylim()[1] - axs[i+1].get_ylim()[0],
                edgecolor = 'red', fill=False))
        axs[0].set_ylabel("acc")
        axs[2].set_xlabel("angle")
        logger.log(f'saving figure to: angle_vs_accuracy_{dataset_name}.png')
        plt.savefig(f"angle_vs_accuracy_{dataset_name}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("SETN")
    parser.add_argument('--data_path',          type=str,   help='Path to dataset')
    parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
    # channels and number-of-cells
    parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
    parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
    parser.add_argument('--channel',            type=int,   help='The number of channels.')
    parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
    parser.add_argument('--select_num',         type=int,   help='The number of selected architectures to evaluate.')
    parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')
    # architecture leraning rate
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
    # log
    parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
    parser.add_argument('--new_save_dir',       type=str,   help='Folder to save checkpoints and log.')

    parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
    parser.add_argument('--rand_seed',          type=int,   help='manual seed')
    parser.add_argument('--resume', type=str, default='', help='The checkpoint path')
    parser.add_argument('--base_epoch',         type=int,   help='initial model load epoch')

    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
    main(args)
