import numpy as np
import torch
from collections import deque

def cache_init(model_kwargs, num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    for j in range(28):
        cache[-1][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1][j] = {}
    for i in range(num_steps):
        cache[i]={}
        for j in range(28):
            cache[i][j] = {}
    cache_dic['cache_type']           = model_kwargs['cache_type']
    cache_dic['cache_index']          = cache_index
    cache_dic['cache']                = cache
    cache_dic['fresh_ratio_schedule'] = model_kwargs['ratio_scheduler']
    cache_dic['fresh_ratio']          = model_kwargs['fresh_ratio']
    cache_dic['fresh_threshold']      = model_kwargs['fresh_threshold']
    cache_dic['force_fresh']          = model_kwargs['force_fresh']
    cache_dic['soft_fresh_weight']    = model_kwargs['soft_fresh_weight']
    cache_dic['flops']                = 0.0
    cache_dic['test_FLOPs']           = model_kwargs['test_FLOPs'] 
    
    cache_dic['cache'][-1]['noise_steps'] = {}
    
    current = {}
    current['num_steps'] = num_steps
    current['num_layers'] = 28

    cache_dic['PRIOR_ERROR_MODELING'] = model_kwargs['PRIOR_ERROR_MODELING']
    if cache_dic['PRIOR_ERROR_MODELING']:
        cache_dic['prior_path']           = model_kwargs['prior_path']
        cache_dic['prior_errors']         = torch.zeros(num_steps)
        cache_dic['prior_cache']          = deque()
        cache_dic['sample_id']            = model_kwargs['sample_id']
        cache_dic['PEM_C']                = model_kwargs['PEM_C']
    cache_dic['DCS_Ns']                   = model_kwargs['DCS_Ns']
    cache_dic['DCS_interval']             = model_kwargs['DCS_interval']
    cache_dic['DCS_weighter']             = model_kwargs['DCS_weighter']
    cache_dic['DCS_error_path']           = model_kwargs['DCS_error_path']
    cache_dic['DCS_timesteps']            = model_kwargs['DCS_timesteps']
    cache_dic['DCS_indices']              = model_kwargs['DCS_indices']
    return cache_dic, current
    