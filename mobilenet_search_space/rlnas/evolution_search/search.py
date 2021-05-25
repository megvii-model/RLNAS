import os
import sys
import time
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from super_model import SuperNetwork
from collections import Counter
from torch.autograd import Variable
from config import config
from test_server import TestClient
import sys
sys.setrecursionlimit(10000)
import argparse
import pickle
import functools
import copy
import time
import functools
print=functools.partial(print,flush=True)
import logging

choice=lambda x:x[np.random.randint(len(x))] if isinstance(x,tuple) else choice(tuple(x))

class EvolutionTrainer(object):
    def __init__(self,log_dir,random=False,refresh=False):
        self.max_epochs=20
        self.select_num=10
        self.population_num=50
        self.m_prob=0.1
        self.crossover_num=25
        self.mutation_num=25
        self.op_flops_dict = pickle.load(open(config.flops_lookup_table, 'rb'))
        self.log_dir=log_dir
        if random:
            #raise NotImplementedError
            self.crossover_num=0
            self.mutation_num=0
            self.checkpoint_name=os.path.join(self.log_dir,'random_search_checkpoint.pkl')
        else:
            self.checkpoint_name=os.path.join(self.log_dir,'search_checkpoint.pkl')

        self.refresh=refresh
        self.tester = TestClient(random=random)
        self.tester.connect()
        self.reset()
        self.w = 0
        self.target_flops = 475 * 1e6
        self.target_params = 7 * 1e6

    def reset(self):
        self.memory=[]
        self.keep_top_k = {self.select_num:[],50:[]}
        self.epoch=0
        self.candidates=[]
        self.vis_dict = {}

    def get_arch_flops(self, cand):
        assert len(cand) == len(config.backbone_info) - 2
        preprocessing_flops = self.op_flops_dict['PreProcessing'][config.backbone_info[0]]
        postprocessing_flops = self.op_flops_dict['PostProcessing'][config.backbone_info[-1]]
        total_flops = preprocessing_flops + postprocessing_flops
        for i in range(len(cand)):
            inp, oup, img_h, img_w, stride = config.backbone_info[i+1]
            op_id = cand[i]
            if op_id >= 0:
                key = config.blocks_keys[op_id]
                total_flops += self.op_flops_dict[key][(inp, oup, img_h, img_w, stride)]
        return total_flops, -1

    def legal(self,cand):
        if len(cand) == 0:
            return False    
        assert isinstance(cand,tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand]={}
        info=self.vis_dict[cand]
        if 'visited' in info:
            return False
        flops = None
        if config.limit_flops:
            if 'flops' not in info:
                info['flops'], info['params']= self.get_arch_flops(cand)
            flops, params = info['flops'], info['params']
            logging.info('flops:{}, params:{}'.format(flops,params))
            identity_num = Counter(cand)[-1]
            # if (config.max_flops is not None and flops > config.max_flops) or (identity_num>config.identity_num):
            if (config.max_flops is not None and flops > config.max_flops):
                return False
        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        logging.info('{} cand = {} flops = {}, params = {}'.format(now, cand, flops, params))
        info['visited']=True
        self.vis_dict[cand]=info
        return True

    def update_top_k(self,candidates,*,k,key,reverse=False):
        assert k in self.keep_top_k
        logging.info('select angle topk......')
        t=self.keep_top_k[k]
        t+=candidates
        t.sort(key=key,reverse=reverse)
        k_ = min(k, 50)
        self.keep_top_k[k]=t[:k_]
    
    def get_topk(self):
        if len(self.keep_top_k[self.select_num]) < 1:
            return None
        topks = []
        for i in range(config.topk):
            topks.append(list(self.keep_top_k[self.select_num][i]))
            logging.info('topk={}'.format(self.keep_top_k[self.select_num]))
        return topks

    def sync_candidates(self, timeout=400):
        while True:
            ok=True
            for cand in self.candidates:
                info=self.vis_dict[cand]
                if 'angle' in info:
                    continue
                ok=False
                if 'test_key' not in info:
                    content = list(cand)
                    now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                    logging.info('{} Sending cand: {}'.format(now, content))
                    info['test_key']=self.tester.send(tuple(content))
            time.sleep(5)
            for cand in self.candidates:
                info=self.vis_dict[cand]
                if 'angle' in info:
                    continue
                key=info['test_key']
                try:
                    logging.info('try to get {}'.format(key))
                    res=self.tester.get(key,timeout=timeout)
                    if res is not None:
                        logging.info('status : {}'.format(res['status']))
                        info.pop('test_key')
                        if 'angle' in res:
                            info['angle']=res['angle']
                except:
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)
                    info.pop('test_key')
            time.sleep(1)
            if ok:
                break

    def stack_random_cand(self,random_func,*,batchsize=10):
        while True:
            cands=[random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand]={}
                info=self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_depth_full_rngs(self, ops, rngs):
        identity_locs = []
        for i, op in enumerate(ops):
            if -1 in op:
                identity_locs.append(i)
        max_identity_num = len(identity_locs)
        identity_num = np.random.randint(max_identity_num+1)
        select_identity = np.random.choice(identity_locs, identity_num, replace=False)
        select_identity = list(select_identity)
        for i in range(len(select_identity)):
            rngs[select_identity[i]] = -1
        return rngs

    def get_random(self,num):
        logging.info('random select ........')
        def get_random_cand():
            rng = []
            for i, ops in enumerate(self.operations):
                if len(ops) == 1:
                    select_op = ops[0]
                else:
                    # if -1 in ops:
                    #     assert(ops[-1]==-1)
                    #     k = np.random.randint(len(ops)-1)
                    # else:
                    #     k = np.random.randint(len(ops))
                    # select_op = ops[k]
                    k = np.random.randint(len(ops))
                    select_op = ops[k]
                rng.append(select_op)
            # rng = self.get_depth_full_rngs(self.operations, rng)
            return tuple(rng)

        cand_iter=self.stack_random_cand(get_random_cand)
        max_iters = num*1000
        while len(self.candidates)<num and max_iters>0: 
            max_iters-=1
            cand=next(cand_iter)
            if not self.legal(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates),num))

        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        logging.info('{} random_num = {}'.format(now, len(self.candidates)))

    def get_mutation(self,k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        max_iters = mutation_num*1000

        def random_func():
            cand = []
            if len(self.keep_top_k[self.select_num]) > 0:
                cand=list(choice(self.keep_top_k[self.select_num]))
            for i in range(len(cand)):
                if np.random.random_sample()<m_prob:
                    k = np.random.randint(len(self.operations[i]))
                    cand[i]=self.operations[i][k]
            return tuple(list(cand))

        cand_iter=self.stack_random_cand(random_func)
        while len(res)<mutation_num and max_iters>0:
            max_iters-=1
            cand=next(cand_iter)
            if not self.legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{} cand={}'.format(len(res),mutation_num,cand))
    
        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        logging.info('{} mutation_num = {}'.format(now, len(res)))
        return res

    def get_crossover(self,k, crossover_num):
        assert k in self.keep_top_k
        logging.info('crossover ......')
        res = []
        max_iters = 1000 * crossover_num
        def random_func():
            if len(self.keep_top_k[self.select_num]) > 0:
                cand1=choice(self.keep_top_k[self.select_num])
                cand2=choice(self.keep_top_k[self.select_num])
                cand = [choice([i,j]) for i,j in zip(cand1,cand2)]
                return tuple(cand)
            else:
                return tuple([])

        cand_iter=self.stack_random_cand(random_func)
        while len(res)<crossover_num and max_iters>0:
            max_iters-=1
            cand=next(cand_iter)
            if not self.legal(cand):
                continue
            res.append(cand)
            logging.info('crossover {}/{} cand={}'.format(len(res),crossover_num,cand))

        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        logging.info('{} crossover_num = {}'.format(now, len(res)))
        return res

    def search(self, operations):
        self.operations = operations
        self.model = SuperNetwork()
        self.layer = len(self.model.features)
        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        logging.info('{} layer = {} population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'\
            .format(now, self.layer, self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - \
                self.mutation_num - self.crossover_num, self.max_epochs))
        self.get_random(self.population_num)

        while self.epoch<self.max_epochs and len(self.candidates) > 0:
            logging.info('epoch = {}'.format(self.epoch))
            self.sync_candidates()
            logging.info('sync finish')

            logging.info('epoch = {}, w = {:.4f}'.format(self.epoch, self.w))
            for cand in self.candidates:
                self.vis_dict[cand]['visited'] = True
            self.update_top_k(self.candidates,k=self.select_num,key=lambda x:self.vis_dict[x]['angle'],reverse=True)
            self.update_top_k(self.candidates,k=50,key=lambda x:self.vis_dict[x]['angle'],reverse=True)

            now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            logging.info('{} epoch = {} : top {} result'.format(now, self.epoch, len(self.keep_top_k[50])))
            for i,cand in enumerate(self.keep_top_k[50]):
                flops = None
                if 'flops' in self.vis_dict[cand]:
                    flops = self.vis_dict[cand]['flops']
                if 'params' in self.vis_dict[cand]:
                    params = self.vis_dict[cand]['params']
                logging.info('No.{} cand={} Top-1 angle = {} flops = {:.2f}M params = {:.2f}'.format(i+1, cand, self.vis_dict[cand]['angle'], flops/1e6, params/1e6))

            crossover = self.get_crossover(self.select_num, self.crossover_num)
            mutation = self.get_mutation(self.select_num, self.population_num-len(crossover), self.m_prob)
            self.candidates = mutation+crossover
            self.get_random(self.population_num)
            self.epoch+=1
        topks = self.get_topk()
        self.reset()
        logging.info('finish!')
        return topks

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-r','--refresh',action='store_true')
    parser.add_argument('--log_path', type=str, default='./log', help='log path')
    parser.add_argument('--seed', type=int, default=1, help='log path')

    args=parser.parse_args()
    refresh=args.refresh
    log_path = args.log_path
    np.random.seed(args.seed)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.log_path, 'search_log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    t = time.time()

    operations = [list(range(config.op_num)) for i in range(config.layers)]
    for i in range(len(operations)):
        if i not in config.stage_last_id and not i == 0:
            operations[i].append(-1)

    trainer=EvolutionTrainer(log_path, refresh=refresh)

    trainer.search(operations)
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
