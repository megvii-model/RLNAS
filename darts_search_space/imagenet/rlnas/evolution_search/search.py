import os
import sys
import time
import glob
import numpy as np
import pickle
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

#from meghair.utils.meta import import_python_source_as_module
#speed_server=import_python_source_as_module('/unsullied/sharefs/muhaoyuan/isilon-share/speed_test/speed_server.py')

from torch.autograd import Variable
from ntools.megtools.classification.config import DpflowProviderMaker, DataproProviderMaker
from IPython import embed
from config import config
from test_server import TestClient
import collections
import sys
sys.setrecursionlimit(10000)
import argparse
import utils
import functools
from thop import profile
from super_model import NetworkImageNet
print=functools.partial(print,flush=True)

choice=lambda x:x[np.random.randint(len(x))] if isinstance(x,tuple) else choice(tuple(x))

class EvolutionTrainer(object):
    def __init__(self,log_dir,*,refresh=False):
        #self.model_for_flops=Network()

        self.log_dir=log_dir
        self.checkpoint_name=os.path.join(self.log_dir,'checkpoint.brainpkl')

        self.refresh=refresh

        self.tester = TestClient()
        self.tester.connect()

        self.memory=[]
        self.candidates=[]
        self.vis_dict={}
        self.keep_top_k = {config.select_num:[],50:[]}
        self.epoch=0
        self.operations = [list(range(config.op_num)) for _ in range(config.edges)]
        self.model = NetworkImageNet()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        info={}
        info['memory']=self.memory
        info['candidates']=self.candidates
        info['vis_dict']=self.vis_dict
        info['keep_top_k']=self.keep_top_k
        info['epoch']=self.epoch
        info['tester']=self.tester.save()
        torch.save(info,self.checkpoint_name)
        logging.info('save checkpoint to {}'.format(self.checkpoint_name))

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info=torch.load(self.checkpoint_name)
        self.memory=info['memory']
        self.candidates = info['candidates']
        self.vis_dict=info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']
        self.tester.load(info['tester'])

        if self.refresh:
            for i,j in self.vis_dict.items():
                for k in ['test_key']:
                    if k in j:
                        j.pop(k)
            self.refresh=False

        logging.info('load checkpoint from {}'.format(self.checkpoint_name))
        return True

    def legal(self,cand):
        assert isinstance(cand,tuple) and len(cand)==(2*config.edges)
        if cand not in self.vis_dict:
            self.vis_dict[cand]={}
        info=self.vis_dict[cand]
        if 'visited' in info:
            return False

        if config.flops_limit is not None:
            if 'flops' not in info:
                normal_cell = cand[:config.edges]
                reduction_cell = cand[config.edges:]
                info['flops'], info['params'] = profile(self.model, config.model_input_size, normal_cell, reduction_cell, device=self.device)
            flops, params = info['flops'], info['params']
            print('flops:{}, params:{}'.format(flops, params))
            if config.max_flops is not None and flops > config.max_flops:
                return False

        self.vis_dict[cand]=info
        info['visited']=True

        return True

    def update_top_k(self,candidates,*,k,key,reverse=False):
        assert k in self.keep_top_k
        logging.info('select ......')
        t=self.keep_top_k[k]
        t+=candidates
        t.sort(key=key,reverse=reverse)
        self.keep_top_k[k]=t[:k]

    def sync_candidates(self):
        while True:
            ok=True
            for cand in self.candidates:
                info=self.vis_dict[cand]
                if 'angle' in info:
                    continue
                ok=False
                if 'test_key' not in info:
                    info['test_key']=self.tester.send(cand)

            self.save_checkpoint()

            for cand in self.candidates:
                info=self.vis_dict[cand]
                if 'angle' in info:
                    continue
                key=info.pop('test_key')

                try:
                    logging.info('try to get {}'.format(key))
                    res=self.tester.get(key,timeout=900)
                    logging.info(res)
                    info['angle']=res['angle']
                    self.save_checkpoint()
                except:
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)

            time.sleep(5)
            if ok:
                break

    def stack_random_cand(self,random_func,*,batchsize=10):
        while True:
            cands=[random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand]={}
                else:
                    continue
                info=self.vis_dict[cand]
            #for cand in cands:
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
        logging.info('random select ........')
        candidates = []
        cand_iter=self.stack_random_cand(lambda:tuple(np.random.randint(config.op_num) for _ in range(2*config.edges)))
        while len(candidates)<num:
            cand=next(cand_iter)
            normal_cand = cand[:config.edges]
            reduction_cand = cand[config.edges:]
            normal_cand = utils.check_cand(normal_cand, self.operations)
            reduction_cand = utils.check_cand(reduction_cand, self.operations)
            cand = normal_cand+reduction_cand
            cand = tuple(cand)
            if not self.legal(cand):
                continue
            candidates.append(cand)
            logging.info('random {}/{}'.format(len(candidates),num))
        logging.info('random_num = {}'.format(len(candidates)))
        return candidates

    def get_mutation(self,k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num*10

        def random_func():
            cand=list(choice(self.keep_top_k[k]))
            for i in range(config.edges):
                if np.random.random_sample()<m_prob:
                    cand[i]=np.random.randint(1, config.op_num)
            return tuple(cand)

        cand_iter=self.stack_random_cand(random_func)
        while len(res)<mutation_num and max_iters>0:
            cand=next(cand_iter)
            normal_cand = cand[:config.edges]
            reduction_cand = cand[config.edges:]
            normal_cand = utils.check_cand(normal_cand, self.operations)
            reduction_cand = utils.check_cand(reduction_cand, self.operations)
            cand = normal_cand + reduction_cand
            cand = tuple(cand)
            if not self.legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res),mutation_num))
            max_iters-=1

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self,k, crossover_num):
        assert k in self.keep_top_k
        logging.info('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num
        def random_func():
            p1=choice(self.keep_top_k[k])
            p2=choice(self.keep_top_k[k])
            return tuple(choice([i,j]) for i,j in zip(p1,p2))
        cand_iter=self.stack_random_cand_crossover(random_func, crossover_num)
        while len(res)<crossover_num:
            try:
                cand=next(cand_iter)
                normal_cand = cand[:config.edges]
                reduction_cand = cand[config.edges:]
                normal_cand = utils.check_cand(normal_cand, self.operations)
                reduction_cand = utils.check_cand(reduction_cand, self.operations)
                cand = normal_cand + reduction_cand
                cand = tuple(cand)
            except Exception as e:
                logging.info(e)
                break
            if not self.legal(cand):
                continue
            res.append(cand)
            logging.info('crossover {}/{}'.format(len(res),crossover_num))

        logging.info('crossover_num = {}'.format(len(res)))
        return res

    def train(self):
        logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(config.population_num, config.select_num, config.mutation_num, config.crossover_num, config.population_num - config.mutation_num - config.crossover_num, config.max_epochs))

        if not self.load_checkpoint():
            self.candidates = self.random_can(config.population_num)
            self.save_checkpoint()

        while self.epoch<config.max_epochs:
            logging.info('epoch = {}'.format(self.epoch))

            self.sync_candidates()

            logging.info('sync finish')

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
                self.vis_dict[cand]['visited'] = True

            self.update_top_k(self.candidates,k=config.select_num,key=lambda x:self.vis_dict[x]['angle'], reverse=True)
            self.update_top_k(self.candidates,k=50,key=lambda x:self.vis_dict[x]['angle'], reverse=True )

            logging.info('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[50])))
            for i,cand in enumerate(self.keep_top_k[50]):
                logging.info('No.{} {} Angle = {} FLOPs = {:.2f}M'.format(i+1, cand, self.vis_dict[cand]['angle'], self.vis_dict[cand]['flops']/1e6))
                #ops = [config.blocks_keys[i] for i in cand]
                ops = [config.blocks_keys[i] for i in cand]
                logging.info(ops)

            mutation = self.get_mutation(config.select_num, config.mutation_num, config.m_prob)
            crossover = self.get_crossover(config.select_num,config.crossover_num)
            rand = self.random_can(config.population_num - len(mutation) -len(crossover))
            self.candidates = mutation+crossover+rand

            self.epoch+=1
            self.save_checkpoint()

        logging.info(self.keep_top_k[config.select_num])
        logging.info('finish!')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-r','--refresh',action='store_true')
    parser.add_argument('--save', type=str, default='log', help='experiment name')
    parser.add_argument('--seed', type=int, default=1, help='experiment name')

    args=parser.parse_args()

    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'search_log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    refresh=args.refresh
    np.random.seed(args.seed)

    t = time.time()

    trainer=EvolutionTrainer(args.save,refresh=refresh)

    trainer.train()
    logging.info('total searching time = {:.2f} hours'.format((time.time()-t)/3600))

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)

