import os
class config:
    host = 'zhangxuanyang.zhangxuanyang.ws2.hh-c.brainpp.cn'

    username = 'admin'
    port = 5672

    exp_name = os.path.dirname(os.path.abspath(__file__))
    exp_name = '-'.join(i for i in exp_name.split(os.path.sep) if i);

    test_send_pipe = exp_name + '-test-send_pipe'
    test_recv_pipe = exp_name + '-test-recv_pipe'

    net_cache = 'model_and_data/checkpoint_epoch_250.pth.tar'
    initial_net_cache = 'model_and_data/checkpoint_epoch_0.pth.tar'


    layers = 8
    edges = 14
    model_input_size_imagenet = (1, 3, 224, 224)

    # Candidate operators
    blocks_keys = [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5'
    ]
    op_num = len(blocks_keys)

    # Operators encoding
    NONE = 0
    MAX_POOLING_3x3 = 1
    AVG_POOL_3x3 = 2
    SKIP_CONNECT = 3
    SEP_CONV_3x3 = 4
    SEP_CONV_5x5 = 5
    DIL_CONV_3x3 = 6
    DIL_CONV_5x5 = 7

    time_limit=None
    #time_limit=0.050
    speed_input_shape=[32,3,224,224]

    flops_limit=None

    max_epochs=20
    select_num = 10
    population_num = 50
    mutation_num = 25
    m_prob = 0.1
    crossover_num = 25


    momentum = 0.7
    eps = 1e-5

    # Enumerate all paths of a single cell
    # paths = [[0, 2, 3, 4, 5], [0, 2, 3, 5], [0, 2, 4, 5], [0, 2, 5], [0, 3, 4, 5], [0, 3, 5], [0, 4, 5], [0, 5],
    #          [1, 2, 3, 4, 5], [1, 2, 3, 5], [1, 2, 4, 5], [1, 2, 5], [1, 3, 4, 5], [1, 3, 5], [1, 4, 5], [1, 5]]
    # Enumerate all paths of a single cell
    paths = [[0, 2, 3, 4, 5], [0, 2, 3, 5], [0, 2, 4, 5], [0, 2, 5], [0, 3, 4, 5], [0, 3, 5], [0, 4, 5], [0, 5],
             [1, 2, 3, 4, 5], [1, 2, 3, 5], [1, 2, 4, 5], [1, 2, 5], [1, 3, 4, 5], [1, 3, 5], [1, 4, 5], [1, 5],
             [0, 2, 3, 4], [0, 2, 4], [0, 3, 4], [0, 4],
             [1, 2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 4],
             [0, 2, 3], [0, 3],
             [1, 2, 3], [1, 3],
             [0, 2],
             [1, 2]]

for i in ['exp_name']:
    print('{}: {}'.format(i,eval('config.{}'.format(i))))
