#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Monday Oct 12th 2020
Last modified on Mondy Feb 1st 2021

@author: Zhuohuang Zhang @ Ph.D. Student at Indiana University Bloomington
"""

import argparse
import os
import librosa
import torch
import numpy as np

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from contextlib import redirect_stdout
from data import AudioDataset, AudioDataLoader
from model import NIMetricNet


parser = argparse.ArgumentParser('Testing/Inference Stage')
parser.add_argument('--model_path', type=str, default='exp/models/best.pth.tar',
                    help='Model to be evaluated. Default: the best model')
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Use GPU or not')
parser.add_argument('--batch_size', default=10, type=int,
                    help='Mini-batch size during inference stage, default set to 10')

# Removing padded audios
def remove_pad(inputs, inputs_lengths):
    """
    Inputs:
        inputs: padded audios with shape B X C X T or B X T
        inputs_lengths: B
    Outputs:
        results: A list with shape B X T_orig
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3: # [B, C, T]
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]
            results.append(input[:length].view(-1).cpu().numpy())
    return results

# Testing stage
def testing(args):

    batch_size = args.batch_size # default is 10
    dataset = 'VOICES' # Specify dataset for test COSINE/VOICES
    test_file = dataset + '_test_audio_file.npy'

    # Load model
    print("Loading model from {}".format(args.model_path))
    with open('eval.log', 'w+') as f:
        f.write("Loading model from {}".format(args.model_path))
        f.write("\n")
    
    model = NIMetricNet.load_model(args.model_path)
    print(model)
    # save to log
    with open('eval.log', 'a+') as f:
        with redirect_stdout(f):
            print(model)
            f.write("\n")

    model.eval() # Turn off for BN during Testing
    if args.use_cuda:
        model.cuda()

    # Testing data
    test_dataset = AudioDataset(test_file, int(batch_size))
    test_loader = AudioDataLoader(test_dataset, batch_size=1)

    # Loading ground truth 
    gt_data = np.load("../data/Y_"+test_file)
    gt_MOS, gt_PESQ, gt_eSTOI, gt_SDR = [], [], [], []
    for tmp_idx in range(len(gt_data)):
        gt_MOS.append(float(gt_data[tmp_idx][0]))
        gt_PESQ.append(float(gt_data[tmp_idx][1]))
        gt_eSTOI.append(float(gt_data[tmp_idx][2]))
        gt_SDR.append(float(gt_data[tmp_idx][3]))
   
    #os.makedirs(args.out_dir, exist_ok=True)

    def write(inputs, filename, fs=16000):
        librosa.output.write_wav(filename, inputs, fs)# norm=True)

    with torch.no_grad():
        #ctt = 0
        MOS, PESQ, eSTOI, SDR = [], [], [], []
        MOS_est, PESQ_est, eSTOI_est, SDR_est = [], [], [], []
        for (i, minibatch) in enumerate(test_loader):
            # Get batch data
            mix_batch, ref_batch, MOS_batch, PESQ_batch, eSTOI_batch, SDR_batch, ilens = minibatch
            if args.use_cuda:
                mix_batch = mix_batch.cuda()
                ref_batch = ref_batch.cuda()
                MOS_batch = MOS_batch.cuda()
                PESQ_batch = PESQ_batch.cuda() 
                eSTOI_batch = eSTOI_batch.cuda() 
                SDR_batch = SDR_batch.cuda()
                ilens = ilens.cuda()
            
            # get network outputs
            est_MOS, est_PESQ, est_eSTOI, est_SDR = model(mix_batch, ilens)
            for tmp in range(args.batch_size):
                MOS.append([float(est_MOS[tmp].item()), float(MOS_batch[tmp].item())])
                PESQ.append([float(est_PESQ[tmp].item()), float(PESQ_batch[tmp].item())])
                eSTOI.append([float(est_eSTOI[tmp].item()), float(eSTOI_batch[tmp].item())])
                SDR.append([float(est_SDR[tmp].item()), float(SDR_batch[tmp].item())])

                MOS_est.append(float(est_MOS[tmp].item()))
                PESQ_est.append(float(est_PESQ[tmp].item()))
                eSTOI_est.append(float(est_eSTOI[tmp].item()))
                SDR_est.append(float(est_SDR[tmp].item()))

    # Save Predicted Scores to file
    os.makedirs('sys_out/est_metric_score/', exist_ok=True)
    print("Saving estimated metric scores to files at 'sys_out/est_metric_score/'")
    with open('eval.log', 'a+') as f:
        f.write("Saving estimated metric scores to files at 'sys_out/est_metric_score/'")
        f.write("\n")

    np.save('sys_out/est_metric_score/est_MOS.npy', MOS)
    np.save('sys_out/est_metric_score/est_PESQ.npy', PESQ)
    np.save('sys_out/est_metric_score/est_eSTOI.npy', eSTOI)
    np.save('sys_out/est_metric_score/est_SDR.npy', SDR)

    # Calculating PEARSON's Coeff, Mean absolute error, root mean squared error
    # Converting to array
    MOS_est   = np.array(MOS_est)
    PESQ_est  = np.array(PESQ_est)
    eSTOI_est = np.array(eSTOI_est)
    SDR_est   = np.array(SDR_est)

    gt_MOS   = np.array(gt_MOS)
    gt_PESQ  = np.array(gt_PESQ)
    gt_eSTOI = np.array(gt_eSTOI)
    gt_SDR   = np.array(gt_SDR)
    gt_SDR[gt_SDR==np.inf] = 36 # set maximum SDR to 36 dB

    # MOS
    MOS_RMSE = np.sqrt(np.mean((gt_MOS-MOS_est)**2))
    MOS_MAE  = np.mean(np.abs(gt_MOS-MOS_est))
    MOS_PCC, _   = pearsonr(gt_MOS, MOS_est)
    # calculate spearman's correlation
    MOS_SRCC, p_MOS = spearmanr(gt_MOS, MOS_est)
    # interpret the significance
    alpha = 0.05
    if p_MOS > alpha:
        print('Samples are uncorrelated (reject H0) for MOS prediction')

    # PESQ
    PESQ_RMSE = np.sqrt(np.mean((gt_PESQ-PESQ_est)**2))
    PESQ_MAE  = np.mean(np.abs(gt_PESQ-PESQ_est))
    PESQ_PCC, _   = pearsonr(gt_PESQ, PESQ_est)
    # calculate spearman's correlation
    PESQ_SRCC, p_PESQ = spearmanr(gt_PESQ, PESQ_est)
    # interpret the significance
    if p_PESQ > alpha:
        print('Samples are uncorrelated (reject H0) for PESQ prediction')

    # eSTOI
    eSTOI_RMSE = np.sqrt(np.mean((gt_eSTOI-eSTOI_est)**2))
    eSTOI_MAE  = np.mean(np.abs(gt_eSTOI-eSTOI_est))
    eSTOI_PCC, _   = pearsonr(gt_eSTOI, eSTOI_est)
    # calculate spearman's correlation
    eSTOI_SRCC, p_eSTOI = spearmanr(gt_eSTOI, eSTOI_est)
    # interpret the significance
    if p_eSTOI > alpha:
        print('Samples are uncorrelated (reject H0) for eSTOI prediction')

    # SDR
    SDR_RMSE = np.sqrt(np.mean((gt_SDR-SDR_est)**2))
    SDR_MAE  = np.mean(np.abs(gt_SDR-SDR_est))
    SDR_PCC, _   = pearsonr(gt_SDR, SDR_est)
    # calculate spearman's correlation
    SDR_SRCC, p_SDR = spearmanr(gt_SDR, SDR_est)
    # interpret the significance
    if p_SDR > alpha:
        print('Samples are uncorrelated (reject H0) for SDR prediction')


    print("-"*85)
    print("MOS has RMSE of {:.3f}, MAE of {:.3f}, PCC of {:.3f}, and SRCC of {:.3f}".format(MOS_RMSE, MOS_MAE, MOS_PCC, MOS_SRCC))
    print("\n")
    print("PESQ has RMSE of {:.3f}, MAE of {:.3f}, PCC of {:.3f}, and SRCC of {:.3f}".format(PESQ_RMSE, PESQ_MAE, PESQ_PCC, PESQ_SRCC))
    print("\n")
    print("eSTOI has RMSE of {:.3f}, MAE of {:.3f}, PCC of {:.3f}, and SRCC of {:.3f}".format(eSTOI_RMSE, eSTOI_MAE, eSTOI_PCC, eSTOI_SRCC))
    print("\n")
    print("SDR has RMSE of {:.3f}, MAE of {:.3f}, PCC of {:.3f}, and SRCC of {:.3f}".format(SDR_RMSE, SDR_MAE, SDR_PCC, SDR_SRCC))
    print("-"*85)
    print("\n")

    with open('eval.log', 'a+') as f:
        f.write("-"*85)
        f.write("\n")
        f.write("MOS has RMSE of {:.3f}, MAE of {:.3f}, PCC of {:.3f}, and SRCC of {:.3f}".format(MOS_RMSE, MOS_MAE, MOS_PCC, MOS_SRCC))
        f.write("\n")
        f.write("PESQ has RMSE of {:.3f}, MAE of {:.3f}, PCC of {:.3f}, and SRCC of {:.3f}".format(PESQ_RMSE, PESQ_MAE, PESQ_PCC, PESQ_SRCC))
        f.write("\n")
        f.write("eSTOI has RMSE of {:.3f}, MAE of {:.3f}, PCC of {:.3f}, and SRCC of {:.3f}".format(eSTOI_RMSE, eSTOI_MAE, eSTOI_PCC, eSTOI_SRCC))
        f.write("\n")
        f.write("SDR has RMSE of {:.3f}, MAE of {:.3f}, PCC of {:.3f}, and SRCC of {:.3f}".format(SDR_RMSE, SDR_MAE, SDR_PCC, SDR_SRCC))
        f.write("\n")
        f.write("-"*85)
        f.write("\n")


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used

    testing(args)

