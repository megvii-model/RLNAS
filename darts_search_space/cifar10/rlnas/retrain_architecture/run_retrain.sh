/usr/bin/python3.6 -m torch.distributed.launch --nproc_per_node=8 retrain.py --auxiliary --save='RLDARTS-ImageNet' --arch='RLDARTS' --init_channels 46
