# -*- coding: utf-8 -*-

# import librosa   #for audio processing
# import IPython.display as ipd
# from sklearn.preprocessing import LabelEncoder

import os

import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct

from sklearn.model_selection import train_test_split
import tensorflow as tf

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


""" Convert data type from NIST SHPERE to .wav
prams
    train_data_path: path to train data
    test_data_path: path to test data
return
    None
"""
def sph2wav(train_data_path, test_data_path):
    sph_files = glob.glob(train_data_path)
    print(len(sph_files),"train utterences")
    for i in sph_files:
        print("FILE: ",i)
        sph = SPHFile(i)
        #sph.write_wav(filename=i.replace(".WAV","_.wav"))
        sample_count = sph.format['sample_count']
        sample_rate = sph.format['sample_rate']
        end_time = sample_count / sample_rate
        j = sph.write_wav(i.replace(".WAV",".tmp"), 0.0, end_time )
        os.remove(i)
        sph.write_wav(j.replace(".tmp",".wav"), 0.0, end_time )
        os.remove(j)

    sph_files_test = glob.glob(test_data_path)
    print(len(sph_files_test),"test utterences")
    for i in sph_files_test:
        print("FILE: ",i)
        sph = SPHFile(i)
        #sph.write_wav(filename=i.replace(".WAV","_.wav"))
        sample_count = sph.format['sample_count']
        sample_rate = sph.format['sample_rate']
        end_time = sample_count / sample_rate
        j = sph.write_wav(i.replace(".WAV",".tmp"), 0.0, end_time )
        os.remove(i)
        sph.write_wav(j.replace(".tmp",".wav"), 0.0, end_time )
        os.remove(j)
    
""" Create quaternion input matrix 
prams
    X: mfcc tensor of data features
return
    Q: quaternion input matrix
"""
def quaternion_input(X):
    Q = []
    for i in range(X.shape[0]):
        y_ft = X[i,:,:]
    
        Dy_t = np.gradient(y_ft)[1]
        DDy_t = np.gradient(Dy_t)[1]
        DDDy_t = np.gradient(DDy_t)[1]
        
        Q_ft = y_ft + Dy_t + DDy_t + DDDy_t
        
        Q.append(Q_ft)
        
    Q = np.array(Q)
        
    return Q


""" Create mfcc or filter-banks coefficinets matrix from .wav audio signals
prams
    signal: sampled audio signal (16KHz)
    sample_rate: rate of sampling
return
    mfcc: Mel-frequency Cepstral coefficient
    filter-banks: 40-dimensional log Mel-filter-bank coefficients
"""
def wav2mfcc(signal, sample_rate):
    #sample_rate, signal = scipy.io.wavfile.read(file_path)  # File assumed to be in the same directory
    #### signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    
    """ Pre-Enphasis"""
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    """ Framing """
    frame_size = 0.025
    frame_stride = 0.01
    
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    frames *= np.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
    
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    
    """ Filter Banks """
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    """ MFCCs """
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    cep_lifter = 22
    
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*

    """ Mean Normalize Filter Banks """
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    
    return filter_banks


""" Create training and validation test sets according to a given splitting ratio
params
    split_ratio: % of data in train set
    quaternion: True if quaternion features matrix is required
return
    train_data: training features and lables set
    val_data: validation features and lables set
"""
def get_data(split_ratio = 0.8, quaternion=False):
    train_audio_path = './dataset/eng_data_subset'
    
    labels=["yes" , "up", "down", "left", "right", "on", "go"] # "no", "off", "stop"
    indexes = {"yes":0, "up":1, "down":2, "left":3, "right":4, "on":5, "go":6, "off":7, "stop":8, "no":9}
    
    all_wave = []
    all_label = []
    for label in labels:
        print("\n[*] Command word: ", label)
        waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
        for wav in tqdm(waves):
            # print(wav)
            # samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
            # resampled_sognal = librosa.resample(samples, sample_rate, 8000)
            file_path = train_audio_path + '/' + label + '/' + wav
            sample_rate, signal = scipy.io.wavfile.read(file_path)
            mfcc_samples = wav2mfcc(signal, sample_rate) # mfcc_sample
            
            y = indexes[label] #i = indexes[label]
            #y = np.full(x.shape[0], fill_value= (i+1))
                
            if(len(signal)== 16000) : 
                all_wave.append(mfcc_samples) # all_wave.append(mfcc_sample)
                all_label.append(y) # all_label.append(label) # label (for sample in samples)
    
    all_wave = np.array(all_wave)
    all_label = np.array(all_label)
    X_train, X_dev, y_train, y_dev = train_test_split(all_wave, all_label, test_size= (1 - split_ratio), shuffle=True)
    
    y_train_hot = tf.keras.utils.to_categorical(y_train, num_classes=len(labels))
    y_dev_hot = tf.keras.utils.to_categorical(y_dev, num_classes=len(labels))
    
    if quaternion:
        X_train = quaternion_input(X_train)
        X_dev = quaternion_input(X_dev)
        
    train_data = [X_train, y_train_hot]
    val_data = [X_dev, y_dev_hot]
    
    return train_data, val_data, len(labels)
