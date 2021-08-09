#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sunday Oct 11th 2020
Last modified on Sunday Oct 11th 2020

@author: Zhuohuang Zhang @ Ph.D. Student at Indiana University Bloomington
"""

import os
import time

import torch
import numpy as np
from scipy.stats import pearsonr

class Trainer(object):
    
    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['train_loader']
        self.cv_loader = data['dev_loader']
        self.model = model
        self.optimizer = optimizer

        # Configs
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # Save/Load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.best_model = args.best_model
        # training info
        self.print_freq = args.print_freq
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Start training epoch {}...".format(epoch + 1))
            with open('debug.train.log', 'a+') as f:
                f.write("Start training epoch {}...".format(epoch + 1))
                f.write("\n")

            self.model.train()  # Turn on for BN
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)
            with open('debug.train.log', 'a+') as f:
                f.write('-' * 85)
                f.write('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
                f.write('-' * 85)
                f.write("\n")

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)
                with open('debug.train.log', 'a+') as f:
                    f.write('Saving checkpoint model to %s' % file_path)
                    f.write("\n")

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)
            with open('debug.train.log', 'a+') as f:
                f.write('-' * 85)
                f.write('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                    'Valid Loss {2:.3f}'.format(
                        epoch + 1, time.time() - start, val_loss))
                f.write('-' * 85)
                f.write("\n")

            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        with open('debug.train.log', 'a+') as f:
                            f.write("No imporvement for 10 epochs, early stopping.")
                            f.write("\n")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                with open('debug.train.log', 'a+') as f:
                    f.write('Learning rate adjusted to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
                    f.write("\n")
                self.halving = False
            self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.best_model)
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)
                with open('debug.train.log', 'a+') as f:
                    f.write("Find better validated model, saving to %s" % file_path)
                    f.write("\n")


    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        mse_loss = torch.nn.MSELoss()

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (minibatch) in enumerate(data_loader):
            mix_batch, ref_batch, MOS_batch, PESQ_batch, eSTOI_batch, SDR_batch, ilens = minibatch
            # padded_mixture, mixture_lengths, padded_source = data
            if self.use_cuda:
                mix_batch = mix_batch.cuda()
                ref_batch = ref_batch.cuda()
                MOS_batch = MOS_batch.cuda()
                PESQ_batch = PESQ_batch.cuda() 
                eSTOI_batch = eSTOI_batch.cuda() 
                SDR_batch = SDR_batch.cuda()
                ilens = ilens.cuda()

            # get network outputs
            est_MOS, est_PESQ, est_eSTOI, est_SDR = self.model(mix_batch, ilens)

            # 10, 1, 12, 0.1, empirically determined loss weights
            MOS_loss = mse_loss(est_MOS.view(-1), MOS_batch.view(-1)).cuda() * 10
            PESQ_loss = mse_loss(est_PESQ.view(-1), PESQ_batch.view(-1)).cuda() * 1
            eSTOI_loss = mse_loss(est_eSTOI.view(-1), eSTOI_batch.view(-1)).cuda() * 12
            SDR_loss = mse_loss(est_SDR.view(-1), SDR_batch.view(-1)).cuda() * 0.1

            loss = MOS_loss + PESQ_loss + eSTOI_loss + SDR_loss
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0: # print after every 'print_freq' mini-batches
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)
                with open('debug.train.log', 'a+') as f:
                    f.write('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)))
                    f.write("\n")

        return total_loss / (i + 1)

