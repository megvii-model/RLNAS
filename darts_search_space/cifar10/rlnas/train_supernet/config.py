import os
class config:
    # Basic configration
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
    op_num=len(blocks_keys)

    # Operators encoding
    NONE = 0
    MAX_POOLING_3x3 = 1
    AVG_POOL_3x3 = 2
    SKIP_CONNECT = 3
    SEP_CONV_3x3 = 4
    SEP_CONV_5x5 = 5
    DIL_CONV_3x3 = 6
    DIL_CONV_5x5 = 7


    # Shrinking configuration
    exp_name = './'
    net_cache = os.path.join(exp_name, 'weight.pt')
    base_net_cache = os.path.join(exp_name, 'base_weight.pt')
    modify_base_net_cache = os.path.join(exp_name, 'weight_0.pt')
    shrinking_finish_threshold = 1000000
    sample_num = 1000
    per_stage_drop_num = 14
    epsilon = 1e-12
    
    # Enumerate all paths of a single cell
    paths = [[0, 2, 3, 4, 5], [0, 2, 3, 5], [0, 2, 4, 5], [0, 2, 5], [0, 3, 4, 5], [0, 3, 5],[0, 4, 5],[0, 5],
             [1, 2, 3, 4, 5], [1, 2, 3, 5], [1, 2, 4, 5], [1, 2, 5], [1, 3, 4, 5], [1, 3, 5],[1, 4, 5],[1, 5]]


    


