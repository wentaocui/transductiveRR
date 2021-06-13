# This code is modified from https://github.com/jakesnell/prototypical-networks

import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate

class transductProtoNet(MetaTemplate):
  def __init__(self, model_func,  alpha1, alpha2, tau, ssl_weight, n_way, n_support):
    super(transductProtoNet, self).__init__(model_func, n_way, n_support)
    self.CrossEntropyLoss = nn.CrossEntropyLoss()
    self.NLLLoss = nn.NLLLoss(reduction='mean')
    self.alpha1 = alpha1
    self.alpha2 = alpha2
    self.tau = tau
    self.ssl_weight = ssl_weight

  def reset_modules(self):
    return

  def set_forward(self,x, is_feature=False):
    z_support, z_query  = self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()

    # represent each query instance as weighted sum of all other query instances (except itself)
    z_query_2d = z_query.reshape(-1, z_query.size(-1))
    query_all_dist = torch.sum((z_query_2d.unsqueeze(1).repeat(1,z_query_2d.size(0),1) - z_query_2d.unsqueeze(0).repeat(z_query_2d.size(0),1,1))**2, dim=-1)
    mask = 1 - torch.eye(z_query_2d.size(0)).cuda()
    query_cross_dist = torch.masked_select(query_all_dist, mask.bool()).reshape(z_query_2d.size(0), -1)
    attention_A = torch.exp(-query_cross_dist*self.tau) / torch.exp(-query_cross_dist*self.tau).sum(-1, keepdim=True)
    z_query_3d = z_query_2d.unsqueeze(0).repeat(z_query_2d.size(0), 1, 1)
    mask_3d = mask.unsqueeze(-1).repeat(1, 1, z_query_2d.size(-1)).bool()
    z_query_3d_masked = torch.masked_select(z_query_3d, mask_3d).reshape(z_query_2d.size(0),z_query_2d.size(0)-1,-1)
    z_query_weighted = attention_A.unsqueeze(-1).expand_as(z_query_3d_masked) * z_query_3d_masked
    z_query_weightedsum = self.alpha1*z_query_weighted.sum(1) + (1-self.alpha1)*z_query_2d

    # represent each support instance as weighted sum of all other query instances.
    # The query instances here are already weighted sum of all other query instances

    z_support_2d = z_support.reshape(-1, z_support.size(-1))
    support_query_dist = torch.sum((z_support_2d.unsqueeze(1).repeat(1, z_query_weightedsum.size(0), 1) - z_query_weightedsum.unsqueeze(0).repeat(z_support_2d.size(0), 1, 1)) ** 2, dim=-1)
    attention_B = torch.exp(-support_query_dist*self.tau) / torch.exp(-support_query_dist*self.tau).sum(-1, keepdim=True)
    z_query_weightedsum_expanded = z_query_weightedsum.unsqueeze(0).repeat(z_support_2d.size(0),1,1)
    z_support_weighted = attention_B.unsqueeze(-1).expand_as(z_query_weightedsum_expanded) * z_query_weightedsum_expanded
    z_support_weightedsum = self.alpha2*z_support_weighted.sum(1) + (1-self.alpha2)*z_support_2d

    # calculate distance between z_support_weightedsum and embedded support instance feature
    z_support_weightedsum_expanded = z_support_weightedsum.unsqueeze(1).repeat(1, z_support_2d.size(0), 1)
    z_support_2d_expanded = z_support_2d.unsqueeze(0).repeat(z_support_2d.size(0), 1, 1)
    support_view_distance = -torch.sum((z_support_weightedsum_expanded - z_support_2d_expanded) ** 2, -1)

    z_proto = z_support_weightedsum.reshape(self.n_way, self.n_support, -1).mean(1)
    neg_query_dist = -euclidean_dist(z_query_weightedsum, z_proto)

    if is_feature == True:
      return neg_query_dist
    else:
      return neg_query_dist, support_view_distance

  def set_forward_loss(self, x):
    # self supervised loss
    y_support_SSL = torch.from_numpy(np.arange( self.n_way*self.n_support ))
    y_support_SSL = y_support_SSL.cuda()
    neg_query_dist, scores_SSL = self.set_forward(x)
    loss_SSL = self.CrossEntropyLoss(scores_SSL, y_support_SSL)

    # query instances cross entropy loss
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
    y_query = y_query.cuda()
    loss_CE = self.CrossEntropyLoss(neg_query_dist, y_query)

    return neg_query_dist, loss_SSL, loss_CE

  def train_loop(self, epoch, train_loader, optimizer):
    print_freq = len(train_loader) // 2
    avg_SSL_loss, avg_CE_loss = 0, 0
    for i, (x,_ ) in enumerate(train_loader):
      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      optimizer.zero_grad()
      _, loss_SSL, loss_CE = self.set_forward_loss(x)
      loss = self.ssl_weight * loss_SSL + loss_CE
      loss.backward()
      optimizer.step()
      avg_SSL_loss = avg_SSL_loss + loss_SSL.item()
      avg_CE_loss = avg_CE_loss + loss_CE.item()
      if (i + 1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | SSL Loss {:f} | CE Loss {:f}'.format(
          epoch, i + 1, len(train_loader), avg_SSL_loss/float(i+1), avg_CE_loss/float(i+1)) )


  def correct(self, x):
    neg_query_dist, loss_SSL, loss_CE = self.set_forward_loss(x)
    y_query = np.repeat(range( self.n_way ), self.n_query )

    topk_scores, topk_labels = neg_query_dist.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)
    return float(top1_correct), len(y_query), loss_CE.item()*len(y_query), loss_SSL.item()*len(y_query)

  def test_loop(self, test_loader, epoch, record = None):
    loss_CE, loss_SSL = 0., 0.
    count = 0
    acc_all = []

    iter_num = len(test_loader)
    for i, (x,_) in enumerate(test_loader):
      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      correct_this, count_this, loss_CE_this, loss_SSL_this = self.correct(x)
      acc_all.append(correct_this/ count_this*100  )
      loss_CE += loss_CE_this
      loss_SSL += loss_SSL_this
      count += count_this

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('----- Epoch %d | acc %.4f%% +- %.4f%% | CE loss %.4f | SSL loss %.4f -----'%(epoch, acc_mean, 1.96* acc_std/np.sqrt(iter_num), loss_CE/count, loss_SSL/count))

    return acc_mean

def euclidean_dist(x, y):
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)

