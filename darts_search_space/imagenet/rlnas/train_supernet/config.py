import os
class config:
    # Basic configration
    layers = 14
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

    


