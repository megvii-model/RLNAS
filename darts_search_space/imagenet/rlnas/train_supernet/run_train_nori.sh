#!/bin/bash
rlaunch --cpu=16 --gpu=8 --memory=100000 --max-wait-time=24h --preemptible=yes -- /usr/bin/python3.6 -m torch.distributed.launch --nproc_per_node=8 train_nori.py --learning_rate=0.1 --epochs=50

