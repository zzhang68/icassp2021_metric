#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sunday Oct 11th 2020
Last modified on Mondy Feb 1st 2021

@author: Zhuohuang Zhang, Ph.D. Student at Indiana University Bloomington
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NIMetricNet(nn.Module):
    def __init__(self, N, W):
        """
        Inputs:
            N: Embedding dimension (i.e., # of filters for Conv-1D)
            W: Window size for Conv-1D based encoder
        """
        super(NIMetricNet, self).__init__()
        # Hyper-parameter
        self.N, self.W = N, W
        
        # Encoder
        # B X T -> B X N X L 
        # B: batch size, L: Number of Frames, i.e. (T-Window)/(Window/2)+1
        self.encoder  = nn.Conv1d(1, self.N, self.W, bias=False, stride=self.W//2) 
        
        # Metric Net
        # CNN layers
        self.cnn1 = nn.Conv2d(1, 16, 3) # input [B, C, H, W]
        self.cnn2 = nn.Conv2d(16, 32, 3)
        self.cnn3 = nn.Conv2d(32, 64, 3)
        self.cnn4 = nn.Conv2d(64, 128, 3)
        # BN
        self.bn1 = nn.BatchNorm2d(16) # BN for each feature channels
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        # avgpooling
        self.avgpool = nn.AvgPool2d(3) # average pooling 3X3

        # Pyramid BiLSTM
        self.metric_rnn_1 = nn.LSTM(input_size = 256, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True) # hidden size of 128*2 for BiLSTM
        self.metric_rnn_2 = nn.LSTM(input_size = 128*2, hidden_size = 64, num_layers = 1, batch_first = True, bidirectional = True) # hidden size of 64*2 for BiLSTM
        self.metric_rnn_3 = nn.LSTM(input_size = 64*2, hidden_size = 32, num_layers = 1, batch_first = True, bidirectional = True) # hidden size of 32*2 for BiLSTM
        
        self.embed_dim = 64 # output from BiLSTM
        # Attention for different targets
        self.att_MOS = torch.nn.MultiheadAttention(self.embed_dim, num_heads = 1) #Inputs: query: (L, N, E), Output: (L,N,E)
        self.att_PESQ = torch.nn.MultiheadAttention(self.embed_dim, num_heads = 1)
        self.att_eSTOI = torch.nn.MultiheadAttention(self.embed_dim, num_heads = 1)
        self.att_SDR = torch.nn.MultiheadAttention(self.embed_dim, num_heads = 1)

        # Output Layer Non-intrusive Metric
        self.dense_MOS_1   = nn.Linear(576, 100)
        self.dense_PESQ_1  = nn.Linear(576, 100)
        self.dense_eSTOI_1 = nn.Linear(576, 100)
        self.dense_SDR_1   = nn.Linear(576, 100)

        self.dense_MOS_2   = nn.Linear(100, 1) # MOS score
        self.dense_PESQ_2  = nn.Linear(100, 1) # PESQ score
        self.dense_eSTOI_2 = nn.Linear(100, 1) # eSTOI score
        self.dense_SDR_2   = nn.Linear(100, 1) # SDR score

        # Output activation function
        self.relu = torch.nn.ReLU()

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mix, ilens):
        """
        Inputs:
            mix: B X T, B: batch size, T: # samples in time domain, (4s * 16000 in the current experiment)
            ilens: B
        Outputs:
            est_MOS
            est_PESQ
            est_eSTOI
            est_SDR
        """
        
        ####################   Speech Encoder   ##########################
        mix_reshape = mix.unsqueeze(1) # reshape mix: B X T -> B X 1 X T
        # Encoding
        mix_enc = self.encoder(mix_reshape) # mix_enc: B X N X L, L is number of frames (i.e., (T-W)/(W/2)+1 = 2T/W-1)
        
        #############################   Metric Net   ####################################
        # Combine est_ref and mix input
        embed_feature = torch.transpose(mix_enc,1,2) # B X L X N = [B, 6399, 256]
        metric_input = embed_feature # cat B X L X N 
        #################### CNN #####################
        # reshape for CNN
        cnn_input = metric_input[:,None,:,:] # -> B X 1 X L X N
        # CNN layers (4 CNNs)
        # Layer 1
        cnn_out = self.cnn1(cnn_input)
        cnn_out = self.bn1(self.relu(cnn_out))
        avg_pool_out = self.avgpool(cnn_out)# [B, 16, 2132, 84]
        
        # Layer 2
        cnn_out = self.cnn2(avg_pool_out)
        cnn_out = self.bn2(self.relu(cnn_out))
        avg_pool_out = self.avgpool(cnn_out)# [B, 32, 710, 27]
        
        # Layer 3
        cnn_out = self.cnn3(avg_pool_out)
        cnn_out = self.bn3(self.relu(cnn_out))
        avg_pool_out = self.avgpool(cnn_out)# [B, 64, 236, 8]
        
        # Layer 4
        cnn_out = self.cnn4(avg_pool_out)
        cnn_out = self.bn4(self.relu(cnn_out))
        cnn_out = self.avgpool(cnn_out) # B X C X T X Feat [B, 128, 78, 2]
        

        # Reshape for BiLSTM input
        cnn_out = torch.transpose(cnn_out,1,2) # B X T X C X Feat
        rnn_in = cnn_out.contiguous().view(cnn_out.size(0),cnn_out.size(1),-1) # B X T X (C X Feat) = B X 78 X 256
        
        ################ Pyramid BiLSTM ################
        # 1st BiLSTM
        metric_rnn_feat, _ = self.metric_rnn_1(rnn_in) # B X L X 256 = B X 78 X 256
        # reshape to reduce the time dimension by half 
        metric_rnn_feat = self.reduce_t_by_half(metric_rnn_feat) # B X L/2 X 256 = B X 39 X 256
        
        # 2nd BiLSTM
        metric_rnn_feat, _ = self.metric_rnn_2(metric_rnn_feat) # B X L/2 X 128 = B X 39 X 128
        # reshape to reduce the time dimension by half
        metric_rnn_feat = self.reduce_t_by_half(metric_rnn_feat) # B X L/4 X 128 = B X 19 X 128

        # 3rd BiLSTM
        metric_rnn_feat, _ = self.metric_rnn_3(metric_rnn_feat) # B X L/4 X 64 = B X 19 X 64
        # reshape to reduce the time dimension by half
        metric_rnn_feat = self.reduce_t_by_half(metric_rnn_feat)  # B X L/8 X 64 = B X 9 X 64
        
        # Attention (self attention)
        att_mos, _ = self.att_MOS(metric_rnn_feat,metric_rnn_feat,metric_rnn_feat)
        att_pesq, _ = self.att_PESQ(metric_rnn_feat,metric_rnn_feat,metric_rnn_feat)
        att_estoi, _ = self.att_eSTOI(metric_rnn_feat,metric_rnn_feat,metric_rnn_feat)
        att_sdr, _ = self.att_SDR(metric_rnn_feat,metric_rnn_feat,metric_rnn_feat) 

        att_mos   = att_mos.view(att_mos.size(0),-1)
        att_pesq  = att_pesq.view(att_pesq.size(0),-1)
        att_estoi = att_estoi.view(att_estoi.size(0),-1)
        att_sdr   = att_sdr.view(att_sdr.size(0),-1) #[12,576]

        # Dense (2 layer)
        fc_mos   = self.relu(self.dense_MOS_1(att_mos)) # B 
        fc_pesq  = self.relu(self.dense_PESQ_1(att_pesq)) # B 
        fc_estoi = self.relu(self.dense_eSTOI_1(att_estoi)) # B 
        fc_sdr   = self.relu(self.dense_SDR_1(att_sdr)) # B
        

        # Quality Estimation
        est_MOS   = self.relu(self.dense_MOS_2(fc_mos)) # B 
        est_PESQ  = self.relu(self.dense_PESQ_2(fc_pesq)) # B 
        est_eSTOI = self.relu(self.dense_eSTOI_2(fc_estoi)) # B 
        est_SDR   = self.dense_SDR_2(fc_sdr) # B
        
        return est_MOS, est_PESQ, est_eSTOI, est_SDR

    # Used for pyramis BiLSTM
    def reduce_t_by_half(self,input_tensor):
        # get input dimension B X T X C
        dims = input_tensor.size()

        output_tensor = torch.zeros(dims[0],dims[1]//2,dims[2],device = input_tensor.get_device())
        for i in range(dims[1]//2):
            output_tensor[:,i,:dims[2]] = input_tensor[:,i,:] + input_tensor[:,i+1,:]

        return output_tensor

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['N'], package['W'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'N': model.N, 'W': model.W,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package
