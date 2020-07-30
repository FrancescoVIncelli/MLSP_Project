__author__ = 'joerg'

# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/


import numpy as np
#from scikits.audiolab import Sndfile
import scipy.io.wavfile as wav
import python_speech_features as sf
import glob

path = "./data/lisa/data/timit/raw/TIMIT/TRAIN/*/*/*_.wav"
wav_files = glob.glob(path)

filename = "./data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SI648_.wav"

# Mel-Frequency Cepstrum Coefficients, default 12
numcep = 40 # 12
# the number of filters in the filterbank.
numfilt = 40

# the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
winlen = 0.025
# the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
winstep = 0.01
# use  first or first+second order derivation
grad = 1


def get_features(filename, numcep, numfilt, winlen, winstep, method=1, quaternion=False):

    #f = Sndfile(filename, 'r')
    #frames = f.nframes
    #samplerate = f.samplerate
    #data = f.read_frames(frames)
    #data = np.asarray(data)
    samplerate, data = wav.read(filename)
    
    # Claculate mfcc
    feat_raw,energy = sf.fbank(data, samplerate,winlen,winstep, nfilt=numfilt)
    feat = np.log(feat_raw)
    feat = sf.dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = sf.lifter(feat,L=22)
    feat = np.asarray(feat)

    #calc log energy
    log_energy = np.log(energy) #np.log( np.sum(feat_raw**2, axis=1) )
    log_energy = log_energy.reshape([log_energy.shape[0],1])

    mat = ( feat - np.mean(feat, axis=0) ) / (0.5 * np.std(feat, axis=0))
    mat = np.concatenate((mat, log_energy), axis=1)

    # Calculate first order derivatives
    # if grad >= 1:
    #     gradf = np.gradient(mat)[0]
    #     mat = np.concatenate((mat, gradf), axis=1)

    # #calc second order derivatives
    # if grad == 2:
    #     grad2f = np.gradient(gradf)[0]
    #     mat = np.concatenate((mat, grad2f), axis=1)
    
    # Calculate 1st-2nd-3rd order derivatives
    if method:
        gradf = np.gradient(mat)[0]
        mat = np.concatenate((mat, gradf), axis=1)
        
        grad2f = np.gradient(gradf)[0]
        mat = np.concatenate((mat, grad2f), axis=1)
        
        grad3f = np.gradient(grad2f)[0]
        mat = np.concatenate((mat, grad3f), axis=1)
    else:
        zerof = np.zeros(shape=mat.shape)
        mat = np.concatenate((mat, zerof), axis=1)
        
        gradf = np.gradient(mat)[0]
        mat = np.concatenate((mat, gradf), axis=1)
        
        grad2f = np.gradient(gradf)[0]
        mat = np.concatenate((mat, grad2f), axis=1)
    
    if quaternion:
        Q_mat = np.reshape(mat, (mat.shape[0],4,mat.shape[1]//4))
        mat = Q_mat
        
    return mat, data, samplerate



