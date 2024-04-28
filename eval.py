import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import _utils
from hparams import hparams
import librosa
import random
import soundfile as sf
import torch
import torch.nn as nn
from hparams import hparams
from torch.utils.data import Dataset,DataLoader
from dataset import TIMIT_Dataset,my_collect
from model_mapping import DNN_Mapping
import os
import dataset


def eval_file_BN(wav_file,model,para):

    noisy_wav,fs = sf.read(wav_file,dtype='int16')
    noisy_wav = noisy_wav.astype('flot32')

    noisy_LPS,noisy_phase = feature_stft(noisy_wav,para.para_stft)

    noisy_LPS = torch.from_numpy(noisy_LPS)

    noisy_LPS_expand = feature_contex(noisy_LPS,para.n_expand)

    model.eval()
    with torch.no_grad():
        enh_LPS = model(x=noisy_LPS_expand,istraining=False)
        model_dic = model.state_dict()
        BN_weight = model_dic['BNlayer.weight'].data
        BN_weight = torch.unsqueeze(BN_weight,dim=0)

        BN_bias = model_dic['BNlayer.bias'].data
        BN_bias = torch.unsqueeze(BN_bias,dim=0)

        BN_mean = model_dic['BNlayer.running_mean'].data
        BN_mean = torch.unsqueeze(BN_mean,dim=0)

        BN_var = model_dic['BNlayer.running_var'].data
        BN_var = torch.unsqueeze(BN_var,dim=0)

        pred_LPS = pred_LPS.numpy()
        enh_mag = np.exp(pred_LPS.T/2)
        enh_phase = noisy_phase[para.n_expand:-para.n_expand,:].T
        enh_spec = enh_mag*np.exp(1j*enh_phase)

        enh_wav = librosa.istft(enh_spec,hop_length=para.para_stft["hop_length"],win_length=para.para_stft["win_length"])
        return enh_wav