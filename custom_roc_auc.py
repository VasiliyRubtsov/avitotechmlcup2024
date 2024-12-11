from numba import njit, prange
import numpy as np

@njit(parallel=True)
def compute_group_auc(y_pred, y_true, group_indices):
    n_groups = len(group_indices) - 1
    user_roc_auc = np.empty(n_groups)
    for i in prange(n_groups):
        start = group_indices[i]
        end = group_indices[i + 1]
        p = y_pred[start:end]
        t = y_true[start:end]
        
        pos_count = np.sum(t == 1)
        neg_count = np.sum(t == 0)
        if pos_count == 0 or neg_count == 0:
            user_roc_auc[i] = np.nan
            continue
            
        order = np.argsort(p)
        sorted_p = p[order]
        sorted_t = t[order]

        n = len(p)
        ranks = np.empty(n)
        i_rank = 0
        while i_rank < n:
            j = i_rank
            while j + 1 < n and sorted_p[j + 1] == sorted_p[i_rank]:
                j += 1
            avg_rank = (i_rank + j) / 2.0 + 1 
            for k in range(i_rank, j + 1):
                ranks[k] = avg_rank
            i_rank = j + 1

        sum_ranks_pos = 0.0
        for k in range(n):
            if sorted_t[k] == 1:
                sum_ranks_pos += ranks[k]

        auc = (sum_ranks_pos - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count)
        user_roc_auc[i] = auc

    return user_roc_auc


def custom_roc_auc(y_pred, y_true, group_sizes):
    group_indices = np.cumsum(np.concatenate(([0], group_sizes)))
    user_roc_auc = compute_group_auc(y_pred, y_true, group_indices)
    return np.nanmean(user_roc_auc)

