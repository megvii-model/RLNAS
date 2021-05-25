import os
class config:
    limit_flops = True
    max_flops = 475 * 1e6
    blocks_keys = [
    'mobilenet_3x3_ratio_3',
    'mobilenet_3x3_ratio_6',
    'mobilenet_5x5_ratio_3',
    'mobilenet_5x5_ratio_6',
    'mobilenet_7x7_ratio_3',
    'mobilenet_7x7_ratio_6'
    ]


    host = 'zhangxuanyang.zhangxuanyang.ws2.hh-c.brainpp.cn'
    # host = 'zhangxuanyang-nj.zhangxuanyang.brw.nj-a.brainpp.cn'

    username = 'admin'
    port = 5672

    exp_name = os.path.dirname(os.path.abspath(__file__))
    exp_name = '-'.join(i for i in exp_name.split(os.path.sep) if i);

    test_send_pipe = exp_name + '-test-send_pipe'
    test_recv_pipe = exp_name + '-test-recv_pipe'

    random_test_send_pipe = 'huyiming_random_test'
    random_test_recv_pipe = 'huyiming_random_test_recv'

    net_cache = './model_and_data/checkpoint_epoch_120.pth.tar'
    # initial_net_cache = './model_and_data/base_weight.pt'
    initial_net_cache = './model_and_data/checkpoint_epoch_0.pth.tar'

    flops_lookup_table = './model_and_data/op_flops_dict.pkl'
    layers = 21
    identity_num = 3
    model_input_size_imagenet = (1, 3, 224, 224)
    stage_last_id=[4,8,12,16,20]
    op_num=len(blocks_keys)
    topk = 1
    epsilon = 1e-12
    backbone_info = [ # inp, oup, img_h, img_w, stride
        (3,     40,     224,    224,    2),     #conv1
        (24,    32,     112,    112,    2),     #stride = 2
        (32,    32,     56,     56,     1),
        (32,    32,     56,     56,     1),
        (32,    32,     56,     56,     1),
        (32,    56,     56,     56,     2),     #stride = 2
        (56,    56,     28,     28,     1),
        (56,    56,     28,     28,     1),   
        (56,    56,     28,     28,     1),
        (56,    112,    28,     28,     2),     #stride = 2
        (112,   112,    14,     14,     1),
        (112,   112,    14,     14,     1),  
        (112,   112,    14,     14,     1),
        (112,   128,    14,     14,     1),
        (128,   128,    14,     14,     1),
        (128,   128,    14,     14,     1),
        (128,   128,    14,     14,     1),
        (128,   256,    14,     14,     2),     #stride = 2
        (256,   256,    7,      7,      1),
        (256,   256,    7,      7,      1),
        (256,   256,    7,      7,      1), 
        (256,   432,    7,      7,      1),
        (432,   1728,   7,      7,      1),     # post_processing
    ]
