# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import utils
import copy

class transductMatchingNet(MetaTemplate):
    def __init__(self, model_func,  alpha1, alpha2, tau, ssl_weight, n_way, n_support):
        super(transductMatchingNet, self).__init__( model_func,  n_way, n_support)

        self.loss_fn    = nn.NLLLoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.FCE = FullyContextualEmbedding(self.feat_dim)
        self.G_encoder = nn.LSTM(self.feat_dim, self.feat_dim, 1, batch_first=True, bidirectional=True)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.tau = tau
        self.ssl_weight = ssl_weight

    def encode_training_set(self, S, G_encoder = None):
        if G_encoder is None:
            G_encoder = self.G_encoder
        out_G = G_encoder(S.unsqueeze(0))[0]
        out_G = out_G.squeeze(0)
        G = S + out_G[:,:S.size(1)] + out_G[:,S.size(1):]
        G_norm = torch.norm(G,p=2, dim =1).unsqueeze(1).expand_as(G)
        G_normalized = G.div(G_norm+ 0.00001) 
        return G, G_normalized

    def get_logprobs(self, f, G, G_normalized, Y_S, FCE = None):        
        if FCE is None:
            FCE = self.FCE
        F = FCE(f, G)
        F_norm = torch.norm(F,p=2, dim =1).unsqueeze(1).expand_as(F)
        F_normalized = F.div(F_norm+ 0.00001) 
        #scores = F.mm(G_normalized.transpose(0,1)) #The implementation of Ross et al., but not consistent with origin paper and would cause large norm feature dominate 
        scores = self.relu( F_normalized.mm(G_normalized.transpose(0,1))  ) *100 # The original paper use cosine simlarity, but here we scale it by 100 to strengthen highest probability after softmax
        softmax = self.softmax(scores)
        logprobs =(softmax.mm(Y_S)+1e-6).log()
        return logprobs

    def set_forward(self, x, is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        # Represent each query and support instance transductively
        if is_feature == True:
            z_support, z_query = self.transductive_encode(z_support, z_query, is_feature)
        else:
            z_support, z_query, scores_SSL = self.transductive_encode(z_support, z_query, is_feature)

        z_support   = z_support.contiguous().view( self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view( self.n_way* self.n_query, -1 )
        G, G_normalized = self.encode_training_set( z_support)

        y_s         = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        Y_S         = Variable( utils.one_hot(y_s, self.n_way ) ).cuda()
        f           = z_query
        logprobs = self.get_logprobs(f, G, G_normalized, Y_S)

        if is_feature == True:
            return logprobs
        else:
            return logprobs, scores_SSL

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        logprobs, scores_SSL = self.set_forward(x)

        # self supervised loss
        y_support_SSL = torch.from_numpy(np.arange(self.n_way * self.n_support)).cuda()
        loss_SSL = self.CrossEntropyLoss(scores_SSL, y_support_SSL)

        return self.loss_fn(logprobs, y_query ), loss_SSL

    def cuda(self):
        super(transductMatchingNet, self).cuda()
        self.FCE = self.FCE.cuda()
        return self

    def transductive_encode(self, z_support, z_query, is_feature):
        # represent each query instance as weighted sum of all other query instances (except itself)
        z_query_2d = z_query.reshape(-1, z_query.size(-1))
        query_all_dist = torch.sum((z_query_2d.unsqueeze(1).repeat(1, z_query_2d.size(0), 1) - z_query_2d.unsqueeze(0).repeat(z_query_2d.size(0), 1, 1))**2, dim=-1)
        mask = 1 - torch.eye(z_query_2d.size(0)).cuda()
        query_cross_dist = torch.masked_select(query_all_dist, mask.bool()).reshape(z_query_2d.size(0), -1)
        attention_A = torch.exp(-query_cross_dist * self.tau) / torch.exp(-query_cross_dist * self.tau).sum(-1, keepdim=True)
        z_query_3d = z_query_2d.unsqueeze(0).repeat(z_query_2d.size(0), 1, 1)
        mask_3d = mask.unsqueeze(-1).repeat(1, 1, z_query_2d.size(-1)).bool()
        z_query_3d_masked = torch.masked_select(z_query_3d, mask_3d).reshape(z_query_2d.size(0), z_query_2d.size(0) - 1,-1)
        z_query_weighted = attention_A.unsqueeze(-1).expand_as(z_query_3d_masked) * z_query_3d_masked
        z_query_weightedsum = self.alpha1 * z_query_weighted.sum(1) + (1 - self.alpha1) * z_query_2d

        # represent each support instance as weighted sum of all other query instances.
        # The query instances here are already weighted sum of all other query instances
        z_support_2d = z_support.reshape(-1, z_support.size(-1))
        support_query_dist = torch.sum((z_support_2d.unsqueeze(1).repeat(1, z_query_weightedsum.size(0),1) - z_query_weightedsum.unsqueeze(0).repeat(z_support_2d.size(0), 1, 1))**2, dim=-1)
        attention_B = torch.exp(-support_query_dist * self.tau) / torch.exp(-support_query_dist * self.tau).sum(-1,keepdim=True)
        z_query_weightedsum_expanded = z_query_weightedsum.unsqueeze(0).repeat(z_support_2d.size(0), 1, 1)
        z_support_weighted = attention_B.unsqueeze(-1).expand_as(z_query_weightedsum_expanded) * z_query_weightedsum_expanded
        z_support_weightedsum = self.alpha2 * z_support_weighted.sum(1) + (1 - self.alpha2) * z_support_2d

        # calculate distance between z_support_weightedsum and embedded support instance feature
        z_support_weightedsum_expanded = z_support_weightedsum.unsqueeze(1).repeat(1, z_support_2d.size(0), 1)
        z_support_2d_expanded = z_support_2d.unsqueeze(0).repeat(z_support_2d.size(0), 1, 1)
        support_view_distance = -torch.sum((z_support_weightedsum_expanded - z_support_2d_expanded) ** 2, -1)

        if is_feature == True:
            return z_support_weightedsum, z_query_weightedsum
        else:
            return z_support_weightedsum, z_query_weightedsum, support_view_distance

    def train_loop(self, epoch, train_loader, optimizer ):
        print_freq = 50

        avg_loss, avg_ssl_loss = 0, 0
        for i, (x,_ ) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss, loss_SSL = self.set_forward_loss( x )
            loss_tot = loss + self.ssl_weight*loss_SSL
            loss_tot.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()
            avg_ssl_loss = avg_ssl_loss + loss_SSL.item()

            if (i+1) % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | CE Loss {:f} | SSL Loss {:f}'.format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_ssl_loss/float(i+1), ))

    def correct(self, x):
        scores, scores_SSL = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

class FullyContextualEmbedding(nn.Module):
    def __init__(self, feat_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim*2, feat_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.c_0 = Variable(torch.zeros(1,feat_dim))
        self.feat_dim = feat_dim
        #self.K = K

    def forward(self, f, G):
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose(0,1)
        K = G.size(0) #Tuna to be comfirmed
        for k in range(K):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)
            r = a.mm(G)
            x = torch.cat((f, r),1)

            h, c = self.lstmcell(x, (h, c))
            h = h + f

        return h
    def cuda(self):
        super(FullyContextualEmbedding, self).cuda()
        self.c_0 = self.c_0.cuda()
        return self

