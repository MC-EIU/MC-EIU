import torch
import torch.nn as nn
import warnings
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from models.networks.multihead_attention import MultiheadAttention
from models.networks.multihead_attention import CrossAttention


class InteractModule(Module):
    def __init__(self, opt):
        super(InteractModule, self).__init__()
        self.inter_attention = MultiheadAttention(embed_dim=opt.hidden_size, num_heads=opt.attention_head,
                                                  dropout=opt.attention_dropout)
        self.hence_attention = MultiheadAttention(embed_dim=opt.hidden_size, num_heads=opt.attention_head,
                                                  dropout=opt.attention_dropout)
        self.opt = opt

    def forward(self, query, key, value, activation='sigmoid'):
        # print(f'query.size is {query.size()}')
        inter_output, _ = self.inter_attention(query, key, value)
        # print(f'inter_output.shape is {inter_output.shape}')
        hence_output, _ = self.hence_attention(query, inter_output, inter_output)
        # print(f'hence_output.shape is {hence_output.shape}')

        # Gate machine
        inter_fusion = inter_output + hence_output
        if activation == 'sigmoid':
            act_function = torch.sigmoid
        elif activation == 'relu':
            act_function = F.relu
        else:
            raise ValueError(f'activation must be Sigmoid or ReLu, but got {activation}')

        assert self.opt.ablation in ['normal', 'gate',
                                     'hence'], f'opt.ablation must be normal, gate, or hence, not be {self.opt.ablation}'

        if self.opt.ablation == 'normal':  # no ablation
            inter_weight = act_function(inter_fusion)
            inter_result = torch.multiply(hence_output, inter_weight)

            residual = query + inter_result

        elif self.opt.ablation == 'gate':  # ablation of gate machine
            residual = query + hence_output

        else:  # ablation of hence_attention
            inter_weight = act_function(inter_output)
            inter_result = torch.multiply(inter_output, inter_weight)

            residual = query + inter_result


        result = []
        for i in range(residual.shape[0]):
            result.append(residual[i])
        residual = torch.cat(result, dim=1)
        return residual
