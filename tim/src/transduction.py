import torch
from tqdm import tqdm
import numpy as np

np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def transductive_encode(z_support, z_query, params):
    n_task = z_support.size(0)
    n_feature = z_support.size(-1)
    n_way = params.eval_n_ways
    n_query = z_query.size(1)
    n_support = z_support.size(1)
    support_shots = z_support.size(1) / n_way

    alpha1 = params.transduct_alpha1
    alpha2 = params.transduct_alpha2
    tau1 = params.transduct_tau

    print("=> Runnning transductive re-representation for {} tasks".format(n_task))
    z_query_weightedsum = torch.zeros_like(z_query)
    z_support_weightedsum = torch.zeros_like(z_support)
    mask = 1 - torch.eye(n_query)
    for task in tqdm(range(n_task)):
        # represent each query instance as weighted sum of all other query instances (except itself)
        query_all_dist = torch.sum((z_query[task].unsqueeze(1).repeat(1,n_query,1) - z_query[task].unsqueeze(0).repeat(n_query,1,1)) ** 2, dim=-1)
        query_cross_dist = torch.masked_select(query_all_dist, mask.bool()).reshape(n_query, -1)
        query_cross_dist = 5 * query_cross_dist / torch.max(query_cross_dist, -1, keepdim=True)[0]
        attention_A = torch.exp(-query_cross_dist * tau1) / torch.exp(-query_cross_dist * tau1).sum(-1, keepdim=True)
        mask_3d = mask.unsqueeze(-1).repeat(1,1,n_feature).bool()
        z_query_3d_masked = torch.masked_select(z_query[task].unsqueeze(0).repeat(n_query,1,1), mask_3d).reshape(n_query, n_query-1, -1)
        z_query_weighted = attention_A.unsqueeze(-1).expand_as(z_query_3d_masked) * z_query_3d_masked
        z_query_weightedsum[task] = alpha1 * z_query_weighted.sum(1) + (1 - alpha1) * z_query[task]

        # represent each support instance as weighted sum of all other query instances.
        # The query instances here are already weighted sum of all other query instances
        support_query_dist = torch.sum((z_support[task].unsqueeze(1).repeat(1,n_query,1) - z_query_weightedsum[task].unsqueeze(0).repeat(n_support,1,1)) ** 2, dim=-1)
        support_query_dist = 5 * support_query_dist / torch.max(support_query_dist, -1, keepdim=True)[0]
        attention_B = torch.exp(-support_query_dist * tau1) / torch.exp(-support_query_dist * tau1).sum(-1, keepdim=True)
        z_query_weightedsum_expanded = z_query_weightedsum[task].unsqueeze(0).repeat(n_support,1,1)
        z_support_weighted = attention_B.unsqueeze(-1).expand_as(z_query_weightedsum_expanded) * z_query_weightedsum_expanded
        z_support_weightedsum[task] = alpha2 * z_support_weighted.sum(1) + (1 - alpha2) * z_support[task]

    return z_support_weightedsum, z_query_weightedsum


def iterative_transductive_encode(z_support, z_query, params):
    n_task = z_support.size(0)
    n_feature = z_support.size(-1)
    n_way = params.eval_n_ways
    n_query = z_query.size(1)
    n_support = z_support.size(1)
    support_shots = z_support.size(1) / n_way

    alpha1 = params.transduct_alpha1
    alpha2 = params.transduct_alpha2
    tau1 = params.transduct_tau

    print("=> Runnning transductive re-representation for {} tasks".format(n_task))
    z_query_weightedsum = torch.zeros_like(z_query)
    z_support_weightedsum = torch.zeros_like(z_support)
    mask = 1 - torch.eye(n_query)
    for task in tqdm(range(n_task)):
        # represent each query instance as weighted sum of all other query instances (except itself)
        query_all_dist = torch.sum((z_query[task].unsqueeze(1).repeat(1,n_query,1) - z_query[task].unsqueeze(0).repeat(n_query,1,1)) ** 2, dim=-1)
        query_cross_dist = torch.masked_select(query_all_dist, mask.bool()).reshape(n_query, -1)
        query_cross_dist = 5 * query_cross_dist / torch.max(query_cross_dist, -1, keepdim=True)[0]
        attention_A = torch.exp(-query_cross_dist * tau1) / torch.exp(-query_cross_dist * tau1).sum(-1, keepdim=True)
        mask_3d = mask.unsqueeze(-1).repeat(1,1,n_feature).bool()
        z_query_3d_masked = torch.masked_select(z_query[task].unsqueeze(0).repeat(n_query,1,1), mask_3d).reshape(n_query, n_query-1, -1)
        z_query_weighted = attention_A.unsqueeze(-1).expand_as(z_query_3d_masked) * z_query_3d_masked
        z_query_weightedsum[task] = alpha1 * z_query_weighted.sum(1) + (1 - alpha1) * z_query[task]

        # represent each support instance as weighted sum of all other query instances.
        # The query instances here are already weighted sum of all other query instances
        support_query_dist = torch.sum((z_support[task].unsqueeze(1).repeat(1,n_query,1) - z_query_weightedsum[task].unsqueeze(0).repeat(n_support,1,1)) ** 2, dim=-1)
        support_query_dist = 5 * support_query_dist / torch.max(support_query_dist, -1, keepdim=True)[0]
        attention_B = torch.exp(-support_query_dist * tau1) / torch.exp(-support_query_dist * tau1).sum(-1, keepdim=True)
        z_query_weightedsum_expanded = z_query_weightedsum[task].unsqueeze(0).repeat(n_support,1,1)
        z_support_weighted = attention_B.unsqueeze(-1).expand_as(z_query_weightedsum_expanded) * z_query_weightedsum_expanded
        z_support_weightedsum[task] = alpha2 * z_support_weighted.sum(1) + (1 - alpha2) * z_support[task]

        for iter in range(params.n_ierative_transduct - 1):
            # represent each query instance as weighted sum of all other query instances (except itself)
            z_query[task] = z_query_weightedsum[task]
            query_all_dist = torch.sum((z_query[task].unsqueeze(1).repeat(1, n_query, 1) - z_query[task].unsqueeze(0).repeat(n_query, 1, 1)) ** 2, dim=-1)
            query_cross_dist = torch.masked_select(query_all_dist, mask.bool()).reshape(n_query, -1)
            query_cross_dist = 5 * query_cross_dist / torch.max(query_cross_dist, -1, keepdim=True)[0]
            attention_A = torch.exp(-query_cross_dist * tau1) / torch.exp(-query_cross_dist * tau1).sum(-1, keepdim=True)
            mask_3d = mask.unsqueeze(-1).repeat(1, 1, n_feature).bool()
            z_query_3d_masked = torch.masked_select(z_query[task].unsqueeze(0).repeat(n_query, 1, 1), mask_3d).reshape(n_query, n_query - 1, -1)
            z_query_weighted = attention_A.unsqueeze(-1).expand_as(z_query_3d_masked) * z_query_3d_masked
            z_query_weightedsum[task] = alpha1 * z_query_weighted.sum(1) + (1 - alpha1) * z_query[task]

            # represent each support instance as weighted sum of all other query instances.
            # The query instances here are already weighted sum of all other query instances
            z_support[task] = z_support_weightedsum[task]
            support_query_dist = torch.sum((z_support[task].unsqueeze(1).repeat(1, n_query, 1) - z_query_weightedsum[task].unsqueeze(0).repeat(n_support, 1, 1)) ** 2, dim=-1)
            support_query_dist = 5 * support_query_dist / torch.max(support_query_dist, -1, keepdim=True)[0]
            attention_B = torch.exp(-support_query_dist * tau1) / torch.exp(-support_query_dist * tau1).sum(-1, keepdim=True)
            z_query_weightedsum_expanded = z_query_weightedsum[task].unsqueeze(0).repeat(n_support, 1, 1)
            z_support_weighted = attention_B.unsqueeze(-1).expand_as(z_query_weightedsum_expanded) * z_query_weightedsum_expanded
            z_support_weightedsum[task] = alpha2 * z_support_weighted.sum(1) + (1 - alpha2) * z_support[task]

    return z_support_weightedsum, z_query_weightedsum


