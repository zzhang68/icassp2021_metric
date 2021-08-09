#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sunday Oct 11th 2020
Last modified on Mondy Feb 1st 2021

@author: Zhuohuang Zhang @ Ph.D. Student at Indiana University Bloomington
"""
import os

from data import AudioDataLoader, AudioDataset
from trainer import Trainer
from model import NIMetricNet
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(
    "Non-intrusive Metric for Subjective and Objective Speech Assessment")


# Network architecture
parser.add_argument('--N', default=256, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--W', default=20, type=int,
                    help='Length of the filters in samples (default: 20)')

# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')

# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')

# save and load model
parser.add_argument('--save_folder', default='exp/models/',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--best_model', default='best.pth.tar',
                    help='Location to save best validation model')

# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')


def main(args):
    # Multi GPUs
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
    
    # data
    dataset = 'VOICES' # Specify dataset for training
    train_file = dataset + '_train_audio_file.npy'
    dev_file   = dataset + '_dev_audio_file.npy'
    batch_size = 24

    # Training Data
    train_dataset = AudioDataset(train_file, int(batch_size)) 
    train_loader = AudioDataLoader(train_dataset, batch_size=1,
                                    shuffle=True,
                                    num_workers=10,
                                    pin_memory=True) # dataloader loads in both X and Y
    # Development Data
    dev_dataset = AudioDataset(dev_file, int(batch_size))
    dev_loader = AudioDataLoader(dev_dataset, batch_size=1,
                                    num_workers=10,
                                    pin_memory=True) # dataloader loads in both X and Y

    data = {'train_loader': train_loader, 'dev_loader': dev_loader}

    # model
    model = NIMetricNet(args.N, args.W)
    print("Model Summary \n")
    print(model)

    with open('debug.train.log', 'w+') as f:
        f.write("Model Summary \n")
        with redirect_stdout(f):
            print(model)
            f.write("\n")

    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    trainer = Trainer(data, model, optimizier, args)
    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    # print input arguments
    print(args)
    # start training
    main(args)

