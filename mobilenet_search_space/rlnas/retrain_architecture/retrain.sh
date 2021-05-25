#!/bin/bash
/usr/bin/python3.6 -m torch.distributed.launch --nproc_per_node=8 retrain.py --save='eval_models' --model_id='2 0 -1 2 2 0 2 4 0 4 4 2 4 2 4 2 0 0 0 2 2'