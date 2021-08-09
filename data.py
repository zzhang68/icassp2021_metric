#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday Oct 9th 2020
Last modified on Friday Oct 9th 2020

@author: Zhuohuang Zhang @ Ph.D. Student at Indiana University Bloomington
"""

import os 
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data


# Dataloader for Training/CV Data
class AudioDataset(data.Dataset):

    def __init__(self, data_file, batch_size, sample_rate=16000, cut_len = 4):
        """
        Inputs:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        train_dir = '../data/'

        super(AudioDataset, self).__init__()
        X_file = 'X_' + data_file 
        Y_file = 'Y_' + data_file 
        mix_file = os.path.join(train_dir, X_file) # noisy mixture for network input
        target_file = os.path.join(train_dir, Y_file) # target labels: subjective ratings, PESQ, eSTOI, SDR, clean speech filename

        print('Loading input data from %s ...' %X_file)
        mix_infos = np.load(mix_file) # noisy filenames

        print('Loading label data from %s ...' %Y_file)
        label_infos = np.load(target_file)

        MOS, PESQ, eSTOI, SDR, ref = map(list, zip(*label_infos)) # subjective ratings, PESQ, eSTOI, SDR, clean speech filename
        
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(mix_infos), start + batch_size)

            minibatch.append([mix_infos[start:end],
                                MOS[start:end],
                                PESQ[start:end],
                                eSTOI[start:end],
                                SDR[start:end],
                                ref[start:end]])
            if end == len(mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """
    Inputs:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Outputs:
        mix_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        refs_pad: B x C x T, torch.Tensor
        MOS: B, torch.Tensor
        PESQ: B, torch.Tensor
        eSTOI: B, torch.Tensor
        SDR: B, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1

    mixtures, refs, MOS, PESQ, eSTOI, SDR = load_mini_batch(batch[0])
    
    # get batch of lengths of input sequences
    ilens = np.array([len(mix) for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    # N x T
    mix_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    # N x T
    ref_pad = pad_list([torch.from_numpy(r).float()
                            for r in refs], pad_value)
    
    return mix_pad, ref_pad, torch.from_numpy(MOS).float(), torch.from_numpy(PESQ).float(), torch.from_numpy(eSTOI).float(), torch.from_numpy(SDR).float(), ilens


# Utility functions
# Loading for mini-batch
# truncate to 4s for every file, cut or pad
def load_mini_batch(batch, trunc_len = 4):
    """
    Each info include wav path and wav duration.
    Outputs:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mix_batch, ref_batch, MOS_batch, PESQ_batch, eSTOI_batch, SDR_batch  = [], [], [], [], [], []
    mix_infos, MOS, PESQ, eSTOI, SDR, ref = batch
    

    # for each utterance
    for cur_mix, cur_MOS, cur_PESQ, cur_eSTOI, cur_SDR, cur_ref in zip(mix_infos, MOS, PESQ, eSTOI, SDR, ref):
        
        # read wav file
        mix, _ = librosa.load(cur_mix,sr=16000)
        ref, _ = librosa.load(cur_ref,sr=16000)

        if len(mix) > trunc_len * 16000:
            mix = mix[:trunc_len*16000] # truncate to 4s
            ref = ref[:trunc_len*16000]

        mix_batch.append(mix)
        ref_batch.append(ref)
        MOS_batch.append(cur_MOS)
        PESQ_batch.append(cur_PESQ)
        eSTOI_batch.append(cur_eSTOI)
        SDR_batch.append(cur_SDR)

    # remove np.inf from SDR
    SDR_batch = np.array(SDR_batch).astype(np.float)

    SDR_batch[SDR_batch == np.inf] = 36 # set to 36 dB SDR

    return mix_batch, ref_batch, np.array(MOS_batch).astype(np.float), np.array(PESQ_batch).astype(np.float), np.array(eSTOI_batch).astype(np.float), SDR_batch

# Padding for mini-batch
# truncate to 4s for every file, cut or pad
def pad_list(xs, pad_value, trunc_len = 4):
    n_batch = len(xs)
    # max_len = max(x.size(0) for x in xs)
    max_len = trunc_len * 16000
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":
    import sys
    dataset = 'VOICES' # or VOICES
    # Generate Training/Testing/Validation Data
    X_data = np.load('../data/' + dataset + '_audio_file.npy')
    Y_data = np.load('../data/' + dataset + '_audio_scores.npy',allow_pickle=True)
    data_len = len(X_data)
    dev_flag = True # Use validation data or not
    

    train_data_file = dataset + '_train_audio_file.npy'
    test_data_file = dataset + '_test_audio_file.npy'
    if dev_flag:
        dev_data_file = dataset + '_dev_audio_file.npy'

    # Split
    if dataset == 'COSINE':
        if dev_flag:
            # Train/Test Set
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_data, Y_data, test_size=0.1, random_state=1) # (i.e., 600*3 files for testing)
            # Train/Dev Set
            X_Train, X_dev, Y_Train, Y_dev = train_test_split(X_Train, Y_Train, test_size=0.1, random_state=1) # 0.1 x 0.9 = 0.09 (i.e., 540*3 files for dev, 4860*3 files for training)
        else:
            # Train/Test Set
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_data, Y_data, test_size=0.1, random_state=1) # (i.e., 5400 files for training, 600 files for testing)
    elif dataset == 'VOICES':
        if dev_flag:
            # Train/Test Set
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_data, Y_data, test_size=0.1, random_state=1) # (i.e., 450*4 files for testing)
            # Train/Dev Set
            X_Train, X_dev, Y_Train, Y_dev = train_test_split(X_Train, Y_Train, test_size=0.1, random_state=1) # 0.1 x 0.9 = 0.09 (i.e., 405*4 files for dev, 3645*4 files for training)
        else:
            # Train/Test Set
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_data, Y_data, test_size=0.1, random_state=1) # (i.e., 4050*4 files for training, 450*4 files for testing)
    # Save
    if dev_flag:
        np.save('../data/' + 'X_' + train_data_file, X_Train)
        np.save('../data/' + 'X_' + test_data_file, X_Test)
        np.save('../data/' + 'X_' + dev_data_file, X_dev)

        np.save('../data/' + 'Y_' + train_data_file, Y_Train)
        np.save('../data/' + 'Y_' + test_data_file, Y_Test)
        np.save('../data/' + 'Y_' + dev_data_file, Y_dev)
    else:
        np.save('../data/' + 'X_' + train_data_file, X_Train)
        np.save('../data/' + 'X_' + test_data_file, X_Test)

        np.save('../data/' + 'Y_' + train_data_file, Y_Train)
        np.save('../data/' + 'Y_' + test_data_file, Y_Test)
    
    #data_file, batch_size = sys.argv[1:3]
    train_file = dataset + '_train_audio_file.npy'
    batch_size = 12
    dataset = AudioDataset(train_file, int(batch_size))
    data_loader = AudioDataLoader(dataset, batch_size=1,
                                  num_workers=10)

    # Sample test on data_loader
    for i, batch in enumerate(data_loader):
        if i%135 == 0:
            mix_batch, ref_batch, MOS_batch, PESQ_batch, eSTOI_batch, SDR_batch, ilens = batch
            print(i)
            print(mix_batch.size())
            print(ref_batch.size())
            print(MOS_batch.size())
            print(ilens)
        if i < 10:
            print(i)
            print(mix_batch)
            print(ref_batch)
            print(MOS_batch)

