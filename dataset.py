import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import _utils
from hparams import hparams
import librosa
import random
import soundfile as sf

class TIMIT_Dataset(Dataset):

    def __init__(self,para):

        self.file_scp = para.file_scp
        self.para_stft = para.para_stft
        self.n_expand = para.n_expand

        files = np.loadtxt(self.file_scp,dtype='str')
        self.clean_files = files[:,1].tolist()
        self.noisy_files = files[:,0].tolist()

        print(len(self.clean_files))


    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_wav, fs = sf.read(self.clean_files[idx], dtype='int16')
        clean_wav = clean_wav.astype('float32')

        noisy_wav, fs = sf.read(self.noisy_files[idx], dtype='int16')
        noisy_wav = noisy_wav.astype('float32')

        clean_LPS, _ = feature_stft(clean_wav, self.para_stft)
        noisy_LPS, _ = feature_stft(noisy_wav, self.para_stft)

        X_train = torch.from_numpy(noisy_LPS)
        Y_train = torch.from_numpy(clean_LPS)

        X_train = feature_stft(X_train, self.n_expand)
        Y_train = Y_train[self.n_expand:-self.n_expand, :]
        return X_train, Y_train


def feature_stft(wav,para):
        spec = librosa.stft(wav,
                            n_fft = para["N_fft"],
                            win_length = para["win_length"],
                            hop_length = para["hop_length"],
                            window = para["window"])

        mag = np.abs(spec)
        LPS = np.log(mag**2)
        phase = np.angle(spec)

        return LPS.T,phase.T


def feature_contex(feature,expand):
    feature = feature.unfold(0,2*expand+1,1)
    feature = feature.transpose(1,2)
    feature = feature.view([-1,(2*expand+1)*feature.shape[-1]])
    return feature


def my_collect(batch):
    batch_X = [item[0] for item in batch]
    batch_Y = [item[1] for item in batch]
    batch_X = torch.cat(batch_X,0)
    batch_Y = torch.cat(batch_Y,0)
    return [batch_X.float(),batch_Y.float()]


if __name__ =='__main__':
    para = hparams()

    m_Dataset = TIMIT_Dataset(para)
    m_DataLoader = torch.utils.data.DataLoader(m_Dataset,batch_size=2,shuffle=True,num_workers=4,collate_fn=my_collect)

    for i_batch,sample_batch in enumerate(m_DataLoader):
        train_X = sample_batch[0]
        train_Y = sample_batch[1]
        print(train_X.shape)
        print(train_Y.shape)