import os,shutil
import sys
import argparse
import pdb

parser = argparse.ArgumentParser("RUN")
parser.add_argument('--exp_name', type=str, default='search')
parser.add_argument('--cpu', type=int, default=8)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--memory', type=int, default=20480)
parser.add_argument('--preemptible', type=str, default='no')
parser.add_argument('--log_path', type=str, default='./log')

args = parser.parse_args()

dirname = args.exp_name
os.makedirs(args.log_path, exist_ok=True)
os.system('rlaunch --preemptible={} --cpu={} --gpu={} --memory={} --max-wait-time=24h -- /usr/bin/python3.5 search.py | tee -a {}/searching.log' .format(args.preemptible, args.cpu, args.gpu, args.memory, args.log_path))
