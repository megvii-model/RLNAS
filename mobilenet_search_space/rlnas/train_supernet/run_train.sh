#!/bin/bash
/usr/bin/python3.6 -m torch.distributed.launch --nproc_per_node=8 train.py --epochs=120