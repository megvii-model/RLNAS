export TORCH_HOME='/data/NAS-ECCV/NAS-Bench-102/test2'

# table 1 
rlaunch --cpu=6 --gpu=1 --memory=20000 --max-wait-time=24h -- bash ./scripts-search/algos/train_supernet.sh cifar10 0 250

rlaunch  --cpu=6 --gpu=1 --memory=100000 --max-wait-time=24h -- bash ./scripts-search/algos/get_angle.sh cifar10 0 249

rlaunch --cpu=6 --gpu=1 --memory=100000 --max-wait-time=24h -- bash ./scripts-search/algos/get_acc.sh cifar10 0 249

rlaunch --cpu=6 --gpu=1 --memory=20000 --max-wait-time=24h -- bash ./scripts-search/algos/cal_correlation.sh cifar10 0 'angle_epoch-249'
