import torch
import numpy as np

np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def transductive_encode(z_support, z_query, params):
    n_feature = z_support.shape[-1]
    n_way = params.meta_val_way
    n_query = z_query.shape[0]
    n_support = z_support.shape[0]
    support_shots = n_support / n_way
    alpha1 = params.transduct_alpha1
    alpha2 = params.transduct_alpha2
    tau1 = params.transduct_tau

    mask = 1 - np.eye(n_query)
    # represent each query instance as weighted sum of all other query instances (except itself)
    query_all_dist = np.sum((np.repeat(z_query[:,None,:],n_query,axis=1) - np.repeat(z_query[None,:,:],n_query,axis=0)) ** 2, axis=-1)
    query_cross_dist = query_all_dist[np.array(mask, dtype=bool)].reshape(n_query,-1)
    query_cross_dist = 5 * query_cross_dist / np.max(query_cross_dist, axis=-1, keepdims=True)
    attention_A = np.exp(-query_cross_dist * tau1) / np.sum(np.exp(-query_cross_dist * tau1), axis=-1, keepdims=True)
    mask_3d = np.array(np.repeat(mask[:,:,None],n_feature,axis=-1), dtype=bool)
    z_query_3d_masked = np.repeat(z_query[None,:,:],n_query,axis=0)[mask_3d].reshape(n_query, n_query-1, -1)
    z_query_weighted = np.repeat(attention_A[:,:,None],n_feature,axis=-1) * z_query_3d_masked
    z_query_weightedsum = alpha1 * np.sum(z_query_weighted, axis=1) + (1 - alpha1) * z_query

    # represent each support instance as weighted sum of all other query instances.
    # The query instances here are already weighted sum of all other query instances
    support_query_dist = np.sum((np.repeat(z_support[:,None,:],n_query,axis=1) - np.repeat(z_query_weightedsum[None,:,:],n_support,axis=0)) ** 2, axis=-1)
    support_query_dist = 5 * support_query_dist / np.max(support_query_dist, axis=-1, keepdims=True)
    attention_B = np.exp(-support_query_dist * tau1) / np.sum(np.exp(-support_query_dist * tau1), axis=-1, keepdims=True)
    z_query_weightedsum_expanded = np.repeat(z_query_weightedsum[None,:,:],n_support,axis=0)
    z_support_weighted = np.repeat(attention_B[:,:,None], n_feature, axis=-1) * z_query_weightedsum_expanded
    z_support_weightedsum = alpha2 * np.sum(z_support_weighted, axis=1) + (1 - alpha2) * z_support

    return z_support_weightedsum, z_query_weightedsum


