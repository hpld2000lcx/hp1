import os
import numpy as np
import random
import scipy.io.wavfile as wav
import librosa
import soundfile as sf
from numpy.linalg import norm
def signal_by_db(speech,noise,snr):
    speech = speech.astype(np.int16)
    noise =noise.astype(np.int16)

    len_speech = speech.shape[0]
    len_noise = noise.shape[0]
    start = random.randint(0,len_noise-len_speech)
    end = start+len_speech

    add_noise=noise[start:end]

    add_noise=add_noise/norm(add_noise)*norm(speech)/(10.0**(0.05*snr))
    mix = speech + add_noise
    return mix

if __name__ == "__main__":
    noise_path = 'C:\\Users\\thunder\Desktop\sound\\noisex-92-master\\noise92'
    noises = ['babble','white']

    clean_wavs = np.loadtxt('C:\\Users\\thunder\Desktop\sound\pytorch\\test.scp',dtype='str').tolist()
    clean_path = 'C:\\Users\\thunder\Desktop\sound\yu3lia4oshu4ju4ku4\data'
    path_noisy= 'C:\\Users\\thunder\Desktop\sound\\noisex-92-master\\oise92'

    snrs = [-5,0,5]

    with open('C:\\Users\\thunder\Desktop\sound\pytorch\\train_DNN_enh.scp','wt') as f:

        for noise in noises :
            print(noise)
            noise_file = os.path.join(noise_path,noise+'.wav')
            noise_data,fs = sf.read(noise_file,dtype='int16')

            for clean_wav in clean_wavs:
                clean_file = os.path.join(clean_path,clean_wav)
                clean_data,fs =sf.read(clean_file,dtype='int16')

                for snr in snrs:
                    noisy_file = os.path.join(path_noisy,noise,str(snr),clean_wav)

                    noisy_path,_ = os.path.split(noisy_file)
                    os.makedirs(noisy_path,exist_ok=True)
                    mix = signal_by_db(clean_data,noise_data,snr)
                    noisy_data = np.asarray(mix,dtype= np.int16)
                    sf.write(noisy_file,noisy_data,fs)
                    f.write('%s %s\n'%(noisy_file,clean_file))



