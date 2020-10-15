# -*- coding: utf-8 -*-

# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : timit_preprocess.py
# Description  : Feature preprocessing for TIMIT dataset
# ******************************************************

#
# Modified by Francesco Vincelli, 2020
#

""" Comments
Do MFCC over all *.wav files and parse label file Use os.walk to iterate all files in a root directory
original phonemes:
phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
mapped phonemes(For more details, you can read the main page of this repo):
phn = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
"""

# ===============
# Import packages

import  os
import  argparse
import  glob
import  sys
import  sklearn
import  numpy                               as np
import  scipy.io.wavfile                    as wav
from    pydub                               import AudioSegment
from    sklearn                             import preprocessing
from    speechvalley.feature.core.calcmfcc  import calc_feat_delta
from    TIMITspeech                         import preprocessWavs as prep_wavs

# [check] # from    speechvalley.utils.utils   import get_num_classes


# ======================================
# Data structures for data preprocessing

## original phonemes
phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

## cleaned phonemes
#phn = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']


# for 'PHN' (phoneme) level target computation
mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', \
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',\
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#',\
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#',\
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
             'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
             'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw',\
             'v', 'w', 'y', 'z', 'zh']

# for 'WRD' (word) level target computation
alphabet = u'abcdefghijklmnopqrstuvwxyz '

char_map = {
    ' ': 0,
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
    'i': 9,
    'j': 10,
    'k': 11,
    'l': 12,
    'm': 13,
    'n': 14,
    'o': 15,
    'p': 16,
    'q': 17,
    'r': 18,
    's': 19,
    't': 20,
    'u': 21,
    'v': 22,
    'w': 23,
    'x': 24,
    'y': 25,
    'z': 26,
    "'": 27
    }

# =====================
# Functions definitions

def clean(word):
    # token = re.compile("[\w-]+|'m|'t|'ll|'ve|'d|'s|\'")
    ## LC ALL & strip fullstop, comma and semi-colon which are not required
    new = word.lower().replace('.', '')
    new = new.replace(',', '')
    new = new.replace(';', '')
    new = new.replace('"', '')
    new = new.replace('!', '')
    new = new.replace('?', '')
    new = new.replace(':', '')
    new = new.replace('-', '')
    return new

def wav2feature(rootdir, save_directory, feat_args, label_args, pad_mode='feat'):
    
    feature_len = feat_args.feat_len
    win_len     = feat_args.win_len
    win_step    = feat_args.win_step
    mode        = feat_args.mode
    
    level       = label_args.level
    seq2seq     = label_args.seq2seq
    
    # feat_dir  = os.path.join(save_directory, level, keywords, mode)
    # label_dir = os.path.join(save_directory, level, keywords, 'label')
    
    # if not os.path.exists(label_dir):
    #     os.makedirs(label_dir)
    # if not os.path.exists(feat_dir):
    #     os.makedirs(feat_dir)
    # count = 0
    
    if pad_mode=='audio':
        max_len = get_audio_maxlen(rootdir) # get_batch_maxlen(wav_files)
    else:
        max_len = None
    
    print("max_len: ", max_len)
    
    feat_list = []
    phoneme_list = []
    
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            fullFilename = os.path.join(subdir, file)
            filenameNoSuffix =  os.path.splitext(fullFilename)[0]
            
            # """
            # Skip 'SA' samples (since thay have too strong accent features)
            filenamesplits = os.path.splitext(file)
            basename = filenamesplits[0]
            if basename[:2] == 'SA':
                continue
            # """
            
            if file.endswith('.wav'):
                """ speechvalley method
                rate = None
                sig = None
                try:                        
                    (rate,sig)= wav.read(fullFilename)
                except ValueError as e:
                    if e.message == "File format 'NIST'... not understood.":
                        print('You should use nist2wav.sh to convert NIST format files to WAV files first, nist2wav.sh is in core folder.')
                        return
                # feat = calcfeat_delta(sig,rate,win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)
                
                feat, mfcc_len = calc_feat_delta(sig, feat_args, rate)
                
                # feat = preprocessing.scale(feat)
                # feat = np.transpose(feat)
                """
                
                #### TIMITspeech method
                feat, mfcc_len = prep_wavs.create_mfcc(fullFilename, feat_args)
                
                if level == 'PHN':
                    labelFilename = filenameNoSuffix + '.PHN'
                    # print("\n[** PHN **] File: ", labelFilename)
                    phenome = []
                    with open(labelFilename,'r') as f:
                        if seq2seq is True:
                            phenome.append(len(phn)) # <start token>
                        for line in f.read().splitlines():
                            s=line.split(' ')[2]
                            if s in mapping.keys():
                                s = mapping[s]
                            p_index = group_phn.index(s) # phn.index(s)
                            phenome.append(p_index)
                        if seq2seq is True:
                            phenome.append(len(phn)+1) # <end token>
                        # print(phenome)
                    phenome = np.array(phenome)

                elif level == 'WRD':
                    labelFilename = filenameNoSuffix + '.WRD'
                    phenome = []
                    sentence = ''
                    with open(labelFilename,'r') as f:
                        for line in f.read().splitlines():
                            s=line.split(' ')[2]
                            sentence += s+' '
                            
                            #### added ####
                            s = clean(s)
                            ####
                            if seq2seq is True:
                                phenome.append(28)
                            for c in s:
                                # if c=="'":
                                #     phenome.append(27)
                                # else:
                                #     phenome.append(ord(c)-96)
                                
                                ## added ##
                                t = char_map[c]
                                phenome.append(t)
                                ####
                            phenome.append(0)

                        phenome = phenome[:-1]
                        if seq2seq is True:
                            phenome.append(29)
                    phenome = np.array(phenome)
                    # print(phenome)
                    print(sentence)

                # count+=1
                # print('file index:',count)
                # if save:
                #     featureFilename = feat_dir + filenameNoSuffix.split('/')[-2]+'-'+filenameNoSuffix.split('/')[-1]+'.npy'
                #     np.save(featureFilename,feat)
                #     labelFilename = label_dir + filenameNoSuffix.split('/')[-2]+'-'+filenameNoSuffix.split('/')[-1]+'.npy'
                #     # print(labelFilename)
                #     np.save(labelFilename,phenome)
                    
                feat_list.append(feat)
                phoneme_list.append(phenome)
                
    return feat_list, phoneme_list


def wav2feature_batch(wav_files, label_files, feat_args, label_args, pad_mode='feat'):
    feat_list = []
    phoneme_list = []
    
    feature_len = feat_args.feat_len
    win_len     = feat_args.win_len
    win_step    = feat_args.win_step
    mode        = feat_args.mode
    
    level       = label_args.level
    seq2seq     = label_args.seq2seq
    
    if pad_mode=='audio':
        max_len = get_batch_maxlen(wav_files)
    else:
        max_len = None
    
    print("max_len: ", max_len)
    
    for i in range(len(wav_files)): #tqdm(range(len(wav_files))):
        label_name = str(label_files[i])
        wav_name = str(wav_files[i])
        if wav_name.endswith('.wav'):
            rate = None
            sig = None
            try:
                if max_len:
                    (rate, sig) = pad_audio_seg(wav_name, max_len)
                else:
                    (rate,sig)= wav.read(wav_name)
                    
            except ValueError as e:
                if e.message == "File format 'NIST'... not understood.":
                    print('You should use nist2wav.sh to convert NIST format files to WAV files first, nist2wav.sh is in core folder.')
                    return
            # feat = calcfeat_delta(sig,rate,win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)
            feat, feat_len = calc_feat_delta(sig, feat_args, rate)
            # feat = preprocessing.scale(feat)
            # feat = np.transpose(feat)

            if level == 'phn':
                phenome = []
                with open(label_name,'r') as f:
                    if seq2seq is True:
                        phenome.append(len(phn)) # <start token>
                    for line in f.read().splitlines():
                        s=line.split(' ')[2]
                        p_index = phn.index(s)
                        phenome.append(p_index)
                    if seq2seq is True:
                        phenome.append(len(phn)+1) # <end token>
                    # print(phenome)
                phenome = np.array(phenome)

            elif level == 'cha':
                # labelFilename = filenameNoSuffix + '.WRD'
                labelFilename = label_name
                phenome = []
                sentence = ''
                with open(labelFilename,'r') as f:
                    for line in f.read().splitlines():
                        s=line.split(' ')[2]
                        sentence += s+' '
                        if seq2seq is True:
                            phenome.append(28)
                        for c in s:
                            if c=="'":
                                phenome.append(27)
                            else:
                                phenome.append(ord(c)-96)
                        phenome.append(0)

                    phenome = phenome[:-1]
                    if seq2seq is True:
                        phenome.append(29)
                # print(phenome)
                # print(sentence)
                
            feat_list.append(feat)
            phoneme_list.append(phenome)
                
    return feat_list, phoneme_list



####
####
def wav2featureBatch(wav_files, feat_args, label_args):
    feat_list = []
    phoneme_list = []
    
    feats_len = []
    labels_len = []
    
    feature_len = feat_args.feat_len
    win_len     = feat_args.win_len
    win_step    = feat_args.win_step
    mode        = feat_args.mode
    
    level       = label_args.level
    seq2seq     = label_args.seq2seq
    
    for i in range(len(wav_files)): #tqdm(range(len(wav_files))):
        wav_name = str(wav_files[i])
        # label_name = str(label_files[i])
        
        # get label_name corresponding to audio filename
        filepath,fname = os.path.split(wav_name)
        ext = str('.'+level)
        label_endName = fname.replace('.wav',ext)
        label_name = os.path.join(filepath,label_endName)
        #
        
        if wav_name.endswith('.wav'):
            rate = None
            sig = None
            try:
                (rate,sig)= wav.read(wav_name)
            except ValueError as e:
                if e.message == "File format 'NIST'... not understood.":
                    print('You should use nist2wav.sh to convert NIST format files to WAV files first, nist2wav.sh is in core folder.')
                    return
            # feat = calcfeat_delta(sig,rate,win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)
            feat, feat_len = calc_feat_delta(sig, feat_args, rate)
            feat = preprocessing.scale(feat)
            # feat = np.transpose(feat)
            feats_len.append(feat.shape[0])
            
            if level == 'PHN':
                phenome = []
                with open(label_name,'r') as f:
                    if seq2seq is True:
                        phenome.append(len(phn)) # <start token>
                    for line in f.read().splitlines():
                        s=line.split(' ')[2]
                        p_index = phn.index(s)
                        phenome.append(p_index)
                    if seq2seq is True:
                        phenome.append(len(phn)+1) # <end token>
                    # print(phenome)
                phenome = np.array(phenome)
                labels_len.append(phenome.shape[0])

            elif level == 'WRD':
                # labelFilename = filenameNoSuffix + '.WRD'
                labelFilename = label_name
                phenome = []
                sentence = ''
                with open(labelFilename,'r') as f:
                    for line in f.read().splitlines():
                        s=line.split(' ')[2]
                        sentence += s+' '
                        if seq2seq is True:
                            phenome.append(28)
                        for c in s:
                            if c=="'":
                                phenome.append(27)
                            else:
                                phenome.append(ord(c)-96)
                        phenome.append(0)

                    phenome = phenome[:-1]
                    if seq2seq is True:
                        phenome.append(29)
                labels_len.append(phenome.shape[0])
                # print(phenome)
                # print(sentence)
            

            feat_list.append(feat)
            phoneme_list.append(phenome)
                
            max_time_step = max(feats_len)
            max_str_len = max(labels_len)
            
    return feat_list, phoneme_list, max_time_step, max_str_len
####
####


def get_batch_maxlen(wav_files):
    max_len = 0
    for wav_name in wav_files:
        (_, sample) = wav.read(wav_name)
        sample_len = len(sample)
        max_len = max(max_len, sample_len)
        
    return max_len

####
def get_wavs_list(root_dir):
    wav_files = glob(
        os.path.join(root_dir,"*/*/*.wav")
        )
    return wav_files

def get_audio_maxlen(root_dir):
    wav_files = get_wavs_list(root_dir)
    max_len = 0
    for wav_name in wav_files:
        (_, sample) = wav.read(wav_name)
        sample_len = len(sample)
        max_len = max(max_len, sample_len)
        
    return max_len
####
def pad_audio_seg(filename, max_len,pad_mode='post'):
    audio_seg = AudioSegment.from_wav(filename)
    # Extract samples vector and frame rate from audio segment
    sample = np.array( audio_seg.get_array_of_samples() ).astype(np.int32)
    rate = audio_seg.frame_rate
    
    sample_len = len(sample)
    pad_len = max_len-sample_len+1
    # milliseconds of silence needed
    pad_ms = 1000*(pad_len / rate)
    silence = AudioSegment.silent(duration=pad_ms)
    
    if pad_mode=='post':
        padded_audio_seg = audio_seg + silence
    elif pad_mode=='pre':
        padded_audio_seg = silence + audio_seg
    elif pad_mode=='split':
        left_len = np.floor(pad_len/2)
        right_len = np.ceil(pad_len/2)
        
        pad_ms_left = 1000*(left_len / rate)
        pad_ms_right = 1000*(right_len / rate)
        silenceL = AudioSegment.silent(duration=pad_ms_left)
        silenceR = AudioSegment.silent(duration=pad_ms_right)
        padded_audio_seg = silenceL + audio_seg + silenceR
    
    # Extract samples vector and frame rate from PADDED audio segment
    padded_sample = np.array( padded_audio_seg.get_array_of_samples() ).astype(np.int32)
    frame_rate = padded_audio_seg.frame_rate
    # Check frame rate of padded signal and of original signal are the same
    try:
        assert(frame_rate == rate)
    except Exception as e:
            print(e)
            
    return (frame_rate, padded_sample)


""" [TO REMOVE] from pydub import AudioSegment
# def pad_features(feat_batch, max_time_len):
#     # feat: (?, time_len, mfcc_len)
#     max_time_len = 0
#     pad_lengths = []
    
#     for feat in feat_batch:
#         max_time_len = max(max_time_len, len(feat))

#     rate=16000
#     pad_len = 778-291
#     pad_ms = 1000*(pad_len / rate)
#     silence = AudioSegment.silent(duration=pad_ms)
#     s = np.array(silence.get_array_of_samples()).astype(np.int32)
    
#     feat_silence, f_len = calcfeat_delta(s, feat_args, rate)
"""
    
    

""" [check if needed] def encode_pad_data(feats, targets)
def encode_pad_data(feats, targets):
    MAX_FEAT_LEN = 778
    MAX_LEN_PHN  = 75
    
    feat_lengths = []
    phn_lengths  = []
    
    feat_list = []
    for feat in feats:
        feat_lengths.append(feat.shape[1])
        if feat.shape[1] < MAX_FEAT_LEN:
            pad_len = MAX_FEAT_LEN - feat.shape[1]
            pad_feat = np.zeros((feat.shape[0], pad_len)).astype(np.float32)
            feat_padded = np.hstack([feat, pad_feat])
            feat_list.append(feat_padded)
        else:
            feat_list.append(feat)
                
    
    phoneme_list = []
    for phn in targets:
        phn_lengths.append(phn.shape[0])
        if phn.shape[0] < MAX_LEN_PHN:
            pad_len = MAX_LEN_PHN - phn.shape[0]
            pad_phn = np.zeros((pad_len)).astype(np.float32)
            phoneme_padded = np.hstack([phn, pad_phn])
            phoneme_list.append(phoneme_padded)
        else:
            phoneme_list.append(phn)
            
    print(len(feat_list), len(phoneme_list))
    
    X = np.array(feat_list).astype(np.float32)
    X = X.transpose(0,2,1)
    
    y = np.array(phoneme_list).astype(np.float32)

    # X_train_t = X_train_t.reshape((X_train_t.shape[0],X_train_t.shape[1],
    #                                3, X_train_t.shape[2]//3))
    return [X, y], [feat_lengths, phn_lengths]
"""


# =========== #
# Script Main #

if __name__ == '__main__':
    # character or phoneme
    """
    parser = argparse.ArgumentParser(prog='timit_preprocess',
                                     description=Script to preprocess timit data)
    parser.add_argument("path", help="Directory where Timit dataset is contained", type=str)
    parser.add_argument("save", help="Directory where preprocessed arrays are to be saved",
                        type=str)
    parser.add_argument("-n", "--name", help="Name of the dataset",
                        choices=['train', 'test'],
                        type=str, default='train')
    parser.add_argument("-l", "--level", help="Level",
                        choices=['cha', 'phn'],
                        type=str, default='cha')
    parser.add_argument("-m", "--mode", help="Mode",
                        choices=['mfcc', 'fbank'],
                        type=str, default='mfcc')
    parser.add_argument('--featlen', type=int, default=13, help='Features length')
    parser.add_argument("--seq2seq", help="set this flag to use seq2seq", action="store_true")

    parser.add_argument("-winlen", "--winlen", type=float,
                        default=0.02, help="specify the window length of feature")

    parser.add_argument("-winstep", "--winstep", type=float,
                        default=0.01, help="specify the window step length of feature")

    args = parser.parse_args()
    
    args = parser.parse_args()
    root_directory = args.path
    save_directory = args.save
    level = args.level
    mode = args.mode
    feature_len = args.featlen
    name = args.name
    seq2seq = args.seq2seq
    win_len = args.winlen
    win_step = args.winstep
    """
    
    root_directory = "./data/lisa/data/timit/raw/TIMIT" # args.path
    save_directory = "./data_set" # args.save
    level = 'phn' # 'phn' # args.level
    mode = 'mfcc' # args.mode
    feature_len = 40 # args.featlen
    name = "TRAIN" # args.name
    seq2seq = False # args.seq2seq
    win_len = 0.025 # args.winlen
    win_step = 0.01 # args.winstep
    
    root_directory = os.path.join(root_directory, name)
    if root_directory == ".":
        root_directory = os.getcwd()
    if save_directory == ".":
        save_directory = os.getcwd()
    if not os.path.isdir(root_directory):
        raise ValueError("Root directory does not exist!")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    
    # =============== Trial args ====================== #
    from    dataclasses                    import dataclass
    
    num_cep     = 40
    num_filt    = 40
    
    pre_emph    = 0.97
    NFFT        = 512
    
    low_freq    = 0
    high_freq   = None
    
    delta_order = 3

    @dataclass
    class mfcc_args:
        win_len:    float
        win_step:   float
        mode:       str
        delta:      int
        feat_len:   int
        num_cep:    int
        num_filt:   int
        pre_emph:   float
        add_energy: bool
        nfft:       int
        low_freq:   int
        high_freq:  int
        
    
    @dataclass
    class target_args:
        level:    str
        seq2seq:  bool
        
    feat_args = mfcc_args(win_len=win_len,      # 0.025
                          win_step=win_step,    # 0.01
                          mode=mode,            # 'mfcc'
                          delta=delta_order,    # 3 (delta (delta (delta)))
                          feat_len=feature_len, #40
                          num_cep=num_cep,      # 40
                          num_filt=num_filt,    # 40
                          pre_emph=pre_emph,    # 0.97
                          add_energy=True,      # append energy term to mfcc
                          low_freq=low_freq,    # 0
                          high_freq=high_freq,  # None
                          nfft=NFFT)            # 512
    
    label_args = target_args(level=level,
                             seq2seq=seq2seq)
    
    
    # ====================== #
    
    feat_list, target_list = wav2feature(rootdir=root_directory,
                                         save_directory=save_directory,
                                         keywords=name,
                                         feat_args=feat_args,
                                         label_args=label_args,
                                         save=False)
    
    train_data_xy, train_data_len = encode_pad_data(feat_list, target_list)
    X, y = train_data_xy
    X_len_list, y_len_list = train_data_len
    
    # ============== #
    # Save .npz file #
    param_set = "mfcc" + "-" + str(feature_len) + "_win" + "-" + str(int(win_len*1000)) + "-" +str(int(win_step*1000))

    train_file_name = save_directory + "/timit_160L_" + "train_" + "xy_" + param_set + ".npz"
    # valid_file_name = datasetDir + "/timit_160L_" + "val_" + "xy_" + param_set + ".npz"
    
    np.savez(train_file_name, x=X, y=y)
    # np.savez(valid_file_name, X_dev_t, y_dev_t)
    
    X_len_t = np.array(X_len_list)
    y_len_t = np.array(y_len_list)
    np.savez("./data_set/data_lengths", x=X_len_t, y=y_len_t)