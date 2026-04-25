import os
import numpy as np
import torch
import random
import sys

def gap_flag(sampled_steps, num_steps):
    flags = [0]*num_steps
    for i in range(len(sampled_steps)-1):
        tmp_index = 1
        for j in range(sampled_steps[i]-1, sampled_steps[i+1],-1):
            flags[j] = tmp_index
            tmp_index += 1
    return flags

def DCS_module(cache_dic, num_steps):
    '''
    CEM: dynamic_caching_strategy

    Given T diffusion timesteps, select Ns full-compute indices (the rest are
    skipped via caching) that minimize the total jump error. The first (index 0)
    and last (index T-1) steps are always included, and consecutive selected
    indices must be separated by a GAP g in [G_min, G_max].

    Solved by dynamic programming:
        dp[t][k] = min total error of a length-k path starting at 0, ending at t.
        dp[t][k] = min_{g in [G_min..G_max]} dp[t-g][k-1] + dp_errors[f"C={g-1}"][t]
    '''
    T = num_steps
    Ns = cache_dic['DCS_Ns']
    INF = float('inf')
    error_path = cache_dic['DCS_error_path']
    errors = torch.load(error_path)  # dict: 'C=1'..'C=9' -> tensor(num_steps)
    dp_errors = {'C=0': np.zeros(T)}
    for C in errors.keys():
        tmp = errors[C].detach().cpu().float().numpy()
        tmp = np.flip(tmp)
        tmp = np.cumsum(tmp)
        if cache_dic['DCS_weighter'] == 'none':
            w = 1
        elif cache_dic['DCS_weighter'] == 'linear':
            w = int(C.split('=')[1])
        elif cache_dic['DCS_weighter'] == 'quadratic':
            w = int(C.split('=')[1]) ** 2
        tmp = tmp * w
        tmp[tmp == 0] = np.nan
        dp_errors[C] = tmp

    G_base = int(round(T / Ns))  # base gap
    C_interval = cache_dic['DCS_interval']  
    MAX_GAP = 10  # = max_error_key (=9) + 1
    G_min = max(1, G_base - C_interval)
    G_max = min(MAX_GAP, G_base + C_interval)
    if G_min > G_max:
        raise ValueError(
            f"DCS_module: empty gap range [{G_min}, {G_max}] "
            f"(G_base={G_base}, C_interval={C_interval}, T={T}, Ns={Ns})"
        )

    err = np.stack(
        [dp_errors[f"C={g - 1}"] for g in range(G_min, G_max + 1)], axis=0
    ).astype(np.float64)

    dp = np.full((T, Ns + 1), INF, dtype=np.float64)
    parent = np.full((T, Ns + 1), -1, dtype=np.int64)
    dp[0, 1] = 0.0

    for k in range(2, Ns + 1):

        for t in range(1, T):
            best = INF  
            best_prev = -1
            for g in range(G_min, G_max + 1):
                prev = t - g
                if prev < 0:
                    break
                prev_cost = dp[prev, k - 1]
                if not np.isfinite(prev_cost):
                    continue
                step_err = err[g - G_min, t]
                if not np.isfinite(step_err):
                    continue
                cand = prev_cost + step_err
                if cand < best:
                    best = cand
                    best_prev = prev
            if best_prev >= 0:
                dp[t, k] = best
                parent[t, k] = best_prev

    if not np.isfinite(dp[T - 1, Ns]):
        raise ValueError(
            f"DCS_module: no feasible schedule found "
            f"(T={T}, Ns={Ns}, G_base={G_base}, C_interval={C_interval}, "
            f"gap_range=[{G_min},{G_max}])."
        )

    indices = [T - 1]
    t, k = T - 1, Ns
    while k > 1:
        t = int(parent[t, k])
        indices.append(t)
        k -= 1
    indices.reverse()

    assert len(indices) == Ns, (len(indices), Ns)
    assert indices[0] == 0 and indices[-1] == T - 1, indices
    for a, b in zip(indices, indices[1:]):
        assert b > a, indices
        gap = b - a
        assert G_min <= gap <= G_max, (a, b, gap, G_min, G_max)

    final_indices = [T - 1 - i for i in indices]  # important!
    print('DCS timesteps:', final_indices)
    return final_indices


if __name__ == "__main__":
    test_dict = {
        'DCS_error_path':'DiT/outputs/priors/priors.pt',
        'DCS_interval':1,
        'DCS_weighter':'quadratic',
    }
    timesteps = 50
    test_dict['DCS_Ns'] = 18
    DCS_module(test_dict, timesteps)
    test_dict['DCS_Ns'] = 14
    DCS_module(test_dict, timesteps)
    test_dict['DCS_Ns'] = 12
    DCS_module(test_dict, timesteps)
    test_dict['DCS_Ns'] = 7
    DCS_module(test_dict, timesteps)
