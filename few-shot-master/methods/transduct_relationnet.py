# This code is modified from https://github.com/floodsung/LearningToCompare_FSL 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import utils

class transductRelationNet(MetaTemplate):
    def __init__(self, model_func,  alpha1, alpha2, tau, ssl_weight, n_way, n_support, loss_type = 'mse'):
        super(transductRelationNet, self).__init__(model_func,  n_way, n_support)

        self.loss_type = loss_type  #'softmax'# 'mse'
        self.relation_module = RelationModule( self.feat_dim , 8, self.loss_type ) #relation net features are not pooled, so self.feat_dim is [dim, w, h] 

        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()  
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.tau = tau
        self.ssl_weight = ssl_weight

    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        # Represent each query and support instance transductively
        if is_feature == True:
            z_support = z_support.contiguous().view(self.n_way, self.n_support, -1)
            z_query   = z_query.contiguous().view(self.n_way, self.n_query, -1)
            z_support, z_query = self.transductive_encode(z_support, z_query, is_feature)
            z_support = z_support.view(self.n_way, self.n_support, *self.feat_dim)
            z_query = z_query.view(self.n_way, self.n_query, *self.feat_dim)
        else:
            z_support, z_query, scores_SSL = self.transductive_encode(z_support, z_query, is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view( self.n_way, self.n_support, *self.feat_dim ).mean(1) 
        z_query     = z_query.contiguous().view( self.n_way* self.n_query, *self.feat_dim )
        
        z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query* self.n_way,1,1,1,1)
        z_query_ext = z_query.unsqueeze(0).repeat( self.n_way,1,1,1,1)
        z_query_ext = torch.transpose(z_query_ext,0,1)
        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relation_pairs = torch.cat((z_proto_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
        relations = self.relation_module(relation_pairs).view(-1, self.n_way)

        if is_feature == True:
            return relations
        else:
            return relations, scores_SSL

    def set_forward_adaptation(self,x,is_feature = True): #overwrite parent function
        assert is_feature == True, 'Finetune only support fixed feature' 
        full_n_support = self.n_support
        full_n_query = self.n_query
        relation_module_clone = RelationModule( self.feat_dim , 8, self.loss_type )
        relation_module_clone.load_state_dict(self.relation_module.state_dict())
 

        z_support, z_query  = self.parse_feature(x,is_feature)
        z_support   = z_support.contiguous()
        set_optimizer = torch.optim.SGD(self.relation_module.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        self.n_support = 3
        self.n_query = 2

        z_support_cpu = z_support.data.cpu().numpy()
        for epoch in range(100):
            perm_id = np.random.permutation(full_n_support).tolist()            
            sub_x = np.array([z_support_cpu[i,perm_id,:,:,:] for i in range(z_support.size(0))])
            sub_x = torch.Tensor(sub_x).cuda()
            if self.change_way:
                self.n_way  = sub_x.size(0)
            set_optimizer.zero_grad()
            y = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
            scores = self.set_forward(sub_x, is_feature = True)
            if self.loss_type == 'mse':
                y_oh = utils.one_hot(y, self.n_way)
                y_oh = Variable(y_oh.cuda())            

                loss =  self.loss_fn(scores, y_oh )
            else:
                y = Variable(y.cuda())
                loss = self.loss_fn(scores, y )
            loss.backward()
            set_optimizer.step()

        self.n_support = full_n_support
        self.n_query = full_n_query
        z_proto     = z_support.view( self.n_way, self.n_support, *self.feat_dim ).mean(1) 
        z_query     = z_query.contiguous().view( self.n_way* self.n_query, *self.feat_dim )

        
        z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query* self.n_way,1,1,1,1)
        z_query_ext = z_query.unsqueeze(0).repeat( self.n_way,1,1,1,1)
        z_query_ext = torch.transpose(z_query_ext,0,1)
        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relation_pairs = torch.cat((z_proto_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
        relations = self.relation_module(relation_pairs).view(-1, self.n_way)

        self.relation_module.load_state_dict(relation_module_clone.state_dict())
        return relations
    def set_forward_loss(self, x):
        y = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))

        scores, scores_SSL = self.set_forward(x)

        # self supervised loss
        y_support_SSL = torch.from_numpy(np.arange(self.n_way * self.n_support)).cuda()
        loss_SSL = self.CrossEntropyLoss(scores_SSL, y_support_SSL)

        if self.loss_type == 'mse':
            y_oh = utils.one_hot(y, self.n_way)
            y_oh = Variable(y_oh.cuda())            

            return self.loss_fn(scores, y_oh ), loss_SSL
        else:
            y = Variable(y.cuda())
            return self.loss_fn(scores, y ), loss_SSL

    def transductive_encode(self, z_support, z_query, is_feature):
        # represent each query instance as weighted sum of all other query instances (except itself)
        z_query_2d = z_query.reshape(-1, z_query.size(-1))
        query_all_dist = torch.sum((z_query_2d.unsqueeze(1).repeat(1, z_query_2d.size(0), 1) - z_query_2d.unsqueeze(0).repeat(z_query_2d.size(0), 1, 1)) ** 2, dim=-1)
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
        support_query_dist = torch.sum((z_support_2d.unsqueeze(1).repeat(1, z_query_weightedsum.size(0),1) - z_query_weightedsum.unsqueeze(0).repeat(z_support_2d.size(0), 1, 1)) ** 2, dim=-1)
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

class RelationConvBlock(nn.Module):
    def __init__(self, indim, outdim, padding = 0):
        super(RelationConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C      = nn.Conv2d(indim, outdim, 3, padding = padding )
        self.BN     = nn.BatchNorm2d(outdim, momentum=1, affine=True)
        self.relu   = nn.ReLU()
        self.pool   = nn.MaxPool2d(2)

        self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

        for layer in self.parametrized_layers:
            backbone.init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self,x):
        out = self.trunk(x)
        return out

class RelationModule(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size, loss_type = 'mse'):        
        super(RelationModule, self).__init__()

        self.loss_type = loss_type
        self.sigmoid = nn.Sigmoid()
        padding = 1 if ( input_size[1] <10 ) and ( input_size[2] <10 ) else 0 # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling

        self.layer1 = RelationConvBlock(input_size[0]*2, input_size[0], padding = padding )
        self.layer2 = RelationConvBlock(input_size[0], input_size[0], padding = padding )

        shrink_s = lambda s: int((int((s- 2 + 2*padding)/2)-2 + 2*padding)/2)

        self.fc1 = nn.Linear( input_size[0]* shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size )
        self.fc2 = nn.Linear( hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        if self.loss_type == 'mse':
            out = self.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)

        return out
