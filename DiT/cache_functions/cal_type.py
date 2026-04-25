# def cal_type(cache_dic, current):
#     '''
#     Determine calculation type for this step
#     '''
#     last_steps = (current['step'] <=2)
#     first_step = (current['step'] == (current['num_steps'] - 1))
#     force_fresh = cache_dic['force_fresh']
#     if not first_step:
#         fresh_interval = cache_dic['cal_threshold']
#     else:
#         fresh_interval = cache_dic['fresh_threshold']

#     if (current['step'] % fresh_interval == 0) or first_step:
#         current['type'] = 'full'
        
#     elif ((current['step'] % fresh_interval) % 2 == 1): #[1,3,5] [2,4,6]
#         current['type'] = 'ToCa'
#     # 'aggressive' 'ToCa' 'FORA' 'Residual'
#     else: 
#         current['type'] = 'aggressive'

def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    if cache_dic['PRIOR_ERROR_MODELING'] or current['step'] in cache_dic['DCS_timesteps']:
        current['type'] = 'full'
    elif cache_dic['DCS_indices'][current['step']]%2 == 1:
        current['type'] = 'ToCa'
    else:
        current['type'] = 'aggressive'
