import os
import math
import torch
import random
import numpy as np

import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

def seed_everything(seed: int): #for deterministic result; currently wav2vec2 model and torch.use_deterministic_algorithms is incompatible
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def length_norm(mat):
    norm_mat = []
    for line in mat:
        temp = line / np.sqrt(sum(np.power(line, 2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat

class Classic_Attention(nn.Module):
    def __init__(self, input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))
         
    def forward(self,inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = F.tanh(lin_out.bmm(v_view).squeeze())
        attention_weights_normalized = F.softmax(attention_weights,1)
        return attention_weights_normalized

class attentive_statistics_pooling(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.attention = Classic_Attention(input_dim, embed_dim)
        
    def weighted_sd(self, inputs, attention_weights, mean):
        el_mat_prod = torch.mul(inputs, attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs, el_mat_prod)
        variance = torch.sum(hadmard_prod, 1) - torch.mul(mean, mean)
        return variance
    
    def forward(self,inputs):
        attention_weights = self.attention(inputs)
        el_mat_prod = torch.mul(inputs, attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        mean = torch.mean(el_mat_prod, 1)
        variance = self.weighted_sd(inputs, attention_weights, mean)
        
        return mean, variance
    
class EmbeddingCompressor(nn.Module):
    def __init__(self):
        super(EmbeddingCompressor, self).__init__()
        # Define a 1D convolution layer
        self.conv1d = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        # Define a fully connected layer to reduce the dimension from 768 to 512
        self.leaky_relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(1)
        self.fc = nn.Linear(768, 512)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
    
    def forward(self, x):
        # Input x shape: (batch_size, 3, 768)
        x = self.conv1d(x)  # After conv1d, shape will be (batch_size, 1, 768)
        x = self.bn(x)
        x = self.leaky_relu(x)
        x = x.squeeze(1)    # Remove the second dimension, shape will be (batch_size, 768)
        x = self.fc(x)      # After fully connected layer, shape will be (batch_size, 512)
        
        return x