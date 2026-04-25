import os
import numpy as np
import torch
import random
import sys

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
    errors = torch.load(error_path) 
    dp_errors = {'C=0': np.zeros(T)}
    for C in errors.keys():
        tmp = errors[C].detach().cpu().float().numpy()
        # tmp = np.cumsum(tmp)
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

    print('DCS timesteps:', np.array(indices))
    return indices


def DCS_module_relaxed_end(cache_dic, num_steps):
    '''
    Same DP and error definitions as DCS_module, but the last selected index is not required to be equal to T-1.

    Motivation: When Ns is small, it may not be possible to reach T-1 exactly when using [G_min, G_max] skip steps. This is still fixed here.

    Returns exactly Ns indices; the last t_last can be less than T-1, but must satisfy the following condition:

    **Strictly less than** the lower bound of candidate skip steps G_min (the same G_min used in the main DP section):

    (T - 1) - t_last < G_min

    For example, when T=50 and G_min=6, t_last is at least 44 (the difference from 49 is 5 < 6); 49 is not required to appear in the result.

    When selecting the endpoint, minimize dp[t_last, Ns] + tail_err(t_last), where tail_err is the virtual edge t_last -> T-1 (defined the same as the main DP edge, landing at T-1). In case of a tie, take the larger t_last.
    '''
    T = num_steps
    Ns = cache_dic['DCS_Ns']
    INF = float('inf')
    error_path = cache_dic['DCS_error_path']
    errors = torch.load(error_path)
    dp_errors = {'C=0': np.zeros(T)}
    for C in errors.keys():
        tmp = errors[C].detach().cpu().float().numpy()
        if cache_dic['DCS_weighter'] == 'none':
            w = 1
        elif cache_dic['DCS_weighter'] == 'linear':
            w = int(C.split('=')[1])
        elif cache_dic['DCS_weighter'] == 'quadratic':
            w = int(C.split('=')[1]) ** 2
        tmp = tmp * w
        tmp[tmp == 0] = np.nan
        dp_errors[C] = tmp

    G_base = int(round(T / Ns))
    C_interval = cache_dic['DCS_interval']
    MAX_GAP = 10
    G_min = max(1, G_base - C_interval)
    G_max = min(MAX_GAP, G_base + C_interval)
    if G_min > G_max:
        raise ValueError(
            f"DCS_module_relaxed_end: empty gap range [{G_min}, {G_max}] "
            f"(G_base={G_base}, C_interval={C_interval}, T={T}, Ns={Ns})"
        )

    err = np.stack(
        [dp_errors[f"C={g - 1}"] for g in range(G_min, G_max + 1)], axis=0
    ).astype(np.float64)

    def tail_err(t_end):
        """从 t_end 一次跳到 T-1 的尾部误差（与主 DP 边同定义，落点为 T-1）。"""
        if t_end >= T - 1:
            return 0.0
        g_tail = (T - 1) - t_end
        if g_tail >= G_min:
            return INF
        key = f"C={g_tail - 1}"
        if key not in dp_errors:
            return np.nan
        v = float(dp_errors[key][T - 1])
        return v if np.isfinite(v) else np.nan

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

    # (T-1) - t < G_min  <=>  t > (T-1) - G_min  <=>  t >= T - G_min
    t_low = max(0, T - G_min)
    best_t = -1
    best_cost = INF
    for t in range(t_low, T):
        base = dp[t, Ns]
        if not np.isfinite(base):
            continue
        te = tail_err(t)
        if not np.isfinite(te):
            continue
        total = base + te
        if total < best_cost - 1e-12 or (
            abs(total - best_cost) <= 1e-12 and t > best_t
        ):
            best_cost = total
            best_t = t

    if best_t < 0:
        raise ValueError(
            f"DCS_module_relaxed_end: no feasible schedule found "
            f"(T={T}, Ns={Ns}, G_base={G_base}, C_interval={C_interval}, "
            f"gap_range=[{G_min},{G_max}], require (T-1)-t_last < G_min, "
            f"allowed_end_range=[{t_low},{T - 1}])."
        )

    indices = [best_t]
    t, k = best_t, Ns
    while k > 1:
        t = int(parent[t, k])
        indices.append(t)
        k -= 1
    indices.reverse()

    assert len(indices) == Ns, (len(indices), Ns)
    assert indices[0] == 0, indices
    t_last = indices[-1]
    assert (T - 1) - t_last < G_min, (t_last, T - 1, G_min)

    for a, b in zip(indices, indices[1:]):
        assert b > a, indices
        gap = b - a
        assert G_min <= gap <= G_max, (a, b, gap, G_min, G_max)

    print('DCS timesteps (relaxed end):', np.array(indices))
    return indices


def DCS_module_interval_gaps(cache_dic, num_steps, gap_rules=[(0,1,(1,))]):
    '''
    gap_rules:
      optional, list of (start, end, allowed_gaps)。
    example:
      gap_rules=[
          (0, 1, (1,)),           # prev==0: gap=1 (i -> i+1)
          (20, 25, (1, 2, 3)),    # prev in [20,25): gap 1,2,3
      ]

    '''
    T = num_steps
    Ns = cache_dic['DCS_Ns']
    INF = float('inf')
    error_path = cache_dic['DCS_error_path']
    errors = torch.load(error_path)
    dp_errors = {'C=0': np.zeros(T)}
    for C in errors.keys():
        tmp = errors[C].detach().cpu().float().numpy()
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

    G_base = int(round(T / Ns))
    C_interval = cache_dic['DCS_interval']
    MAX_GAP = 10
    G_min = max(1, G_base - C_interval)
    G_max = min(MAX_GAP, G_base + C_interval)
    if G_min > G_max:
        raise ValueError(
            f"DCS_module_interval_gaps: empty default gap range [{G_min}, {G_max}] "
            f"(G_base={G_base}, C_interval={C_interval}, T={T}, Ns={Ns})"
        )

    # err_full[c, t] = dp_errors["C=c"][t]，对应 gap = c + 1
    err_full = np.stack(
        [dp_errors[f"C={c}"] for c in range(0, MAX_GAP)], axis=0
    ).astype(np.float64)

    def allowed_gaps_for_prev(prev):
        if gap_rules:
            for lo, hi, gaps in gap_rules:
                if lo <= prev < hi:
                    out = []
                    for g in gaps:
                        g = int(g)
                        if 1 <= g <= MAX_GAP and f"C={g - 1}" in dp_errors:
                            out.append(g)
                    return frozenset(out)
        return frozenset(range(G_min, G_max + 1))

    allowed_by_prev = [allowed_gaps_for_prev(p) for p in range(T)]

    dp = np.full((T, Ns + 1), INF, dtype=np.float64)
    parent = np.full((T, Ns + 1), -1, dtype=np.int64)
    dp[0, 1] = 0.0

    for k in range(2, Ns + 1):
        for t in range(1, T):
            best = INF
            best_prev = -1
            lo_prev = max(0, t - MAX_GAP)
            for prev in range(lo_prev, t):
                g = t - prev
                if g not in allowed_by_prev[prev]:
                    continue
                prev_cost = dp[prev, k - 1]
                if not np.isfinite(prev_cost):
                    continue
                step_err = err_full[g - 1, t]
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
            f"DCS_module_interval_gaps: no feasible schedule found "
            f"(T={T}, Ns={Ns}, G_base={G_base}, C_interval={C_interval}, "
            f"default_gap_range=[{G_min},{G_max}], gap_rules={gap_rules})."
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
        assert gap in allowed_by_prev[a], (a, b, gap, allowed_by_prev[a])

    print('DCS timesteps (interval gaps):', np.array(indices))
    return indices


if __name__ == "__main__":
    # test
    test_dict = {
        'DCS_error_path':'Wan21/outputs/priors/priors.pt',
        'DCS_interval':1,
        'DCS_weighter':'quadratic',
    }
    timesteps = 50
    test_dict['DCS_Ns'] = 9
    DCS_module(test_dict, timesteps)
    DCS_module_interval_gaps(test_dict, timesteps, [(0,1,(1,))])
    test_dict['DCS_Ns'] = 7 # extra acceleration ratio
    try:
        DCS_module(test_dict, timesteps)
    except ValueError as e:
        # print('DCS_module (expected may fail for small Ns):', e)
        DCS_module_relaxed_end(test_dict, timesteps)

