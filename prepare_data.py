# -*- coding: utf-8 -*-

# =============== #
# Import packages #

from __future__ import print_function
import numpy as np

import os
# import klepto                                                       # Save pre-processed data
#from utility import print_line                                       # clearer stdout printing funtion

from preprocessing_TIMIT.targets import get_target, get_timit_dict     # Create target variables

from preprocessing_TIMIT.features import get_features                 # Extract input features
                                                                      # for model training

from preprocessing_TIMIT.speakers import get_speaker_lists            # (TODO)

from tqdm import tqdm

# Set the path to the root directory containing the TIMIT dataset
ROOT_DATA_DIR = "./data/lisa/data/timit/raw/TIMIT"


# Check that TRAIN and TEST data folder are not empty
if "TEST/" in os.listdir(ROOT_DATA_DIR):
    raise Warning("TIMIT TEST data is missing")
if "TRAIN/" in os.listdir(ROOT_DATA_DIR):
    raise Warning("TIMIT TRAIN data is missing")

# Location of the target data set folder
datasetDir = "./data_set"



# ===================== #
# Funcrtion definitions #


def get_data_maxlen(root_data_dir):
    numcep = 40     # Mel-Frequency Cepstrum Coefficients, default 12.
    numfilt = 40    # Number of filters in the filterbank, default 26.
    
    winlen = 0.025  # Length of the analysis window in seconds. Default is 0.025s (25 milliseconds).
    winstep = 0.01  # Step between successive windows in seconds. Default is 0.01s (10 milliseconds).
    method = 1      # If 1: compute derivative up to 3rd order. If 0, derivative up to 2nd order.\
                    # Default is 1
                    
    size_dict = {}
    
    dirlist = ['DR5', 'DR6', 'DR7', 'DR3', 'DR2', 'DR1', 'DR4', 'DR8']
    for d in dirlist:
        # print("\nDirectory: ", d)
        path = os.path.join(ROOT_DATA_DIR,os.path.join("TRAIN",d))
        
        for dirName, subdirList, fileList in tqdm( os.walk(path) , ncols=100,desc=d):
            # [check] # print('Found directory: %s' % dirName)
    
            path,folder_name = os.path.split(dirName)
            # [check] # print('Speaker: ' + folder_name)
            if folder_name.__len__() >= 1:
                temp_name = ""
                for fname in sorted(fileList):
                    name = fname.split(".")[0]
                    if name != temp_name:
                        temp_name = name
                        # [check] # print('\t%s' % dirName+"/"+name)
                        #wav_location = dirName+"/"+name+".WAV"
                        wav_location = dirName+"/"+name+".wav"
                        phn_location = dirName+"/"+name+".PHN"
                        feat, sample, rate = get_features(wav_location, numcep, numfilt, winlen, winstep, method,quaternion=True)
                        
                        if len(sample) == 14644:
                            print("shortest: ", wav_location)
                        if len(sample) == 124621:
                            print("longest: ", wav_location)
                        
                        if len(feat) in size_dict:
                            size_dict[len(feat)].append(len(sample))
                        else:
                            size_dict[len(feat)]=[]
                            size_dict[len(feat)].append(len(sample))
                            
    return size_dict
                            
                        
def save_data_to_tensor(X, y, data_class, param_set, path=datasetDir):
    if not os.path.exists(path):
        os.makedirs(path)
        
    filename = "timit_" + data_class + "xy_" + param_set + ".npy"
    
    if not os.path.exists(path):
        os.makedirs(path)
    target_path = os.path.join(path, filename)
        
    X_arr = np.array(X).astype(np.float32)
    y_arr = np.array(y).astype(np.float32)
    
    data_set = {'X':X_arr, 'y':y_arr}
    np.save(target_path, data_set)
  
    
def load_train_data(dir_list=dirlist, path=datasetDir):
    X_train = []
    y_train = []
    X_dev = []
    y_dev = []
    
    for d in dirlist:
        # print("\nDirectory: ", d)
        source_path = os.path.join(path,d)
        for dirName, _, fileList in os.walk(source_path):
            for fname in sorted(fileList):
                if fname.find('train'):
                    print(os.path.join(source_path, fname))
                    train_data = np.load(os.path.join(source_path, fname), allow_pickle=True)
                elif fname.find('val'):
                    print(os.path.join(source_path, fname))
                    valid_data = np.load(os.path.join(source_path, fname), allow_pickle=True)
                else:
                    assert False, "wrong name format"
                    
            X_train_tmp = train_data.item().get('X')
            y_train_tmp = train_data.item().get('y')
            
            X_dev_tmp = train_data.item().get('X')
            y_dev_tmp = train_data.item().get('y')
            
            print(len(X_train_tmp), "-", len(y_train_tmp))
            print(len(X_dev_tmp), "-", len(y_dev_tmp))
            
            X_train.append(X_train_tmp)
            y_train.append(y_train_tmp)
            X_dev.append(X_dev_tmp)
            y_dev.append(y_dev_tmp)
    
    return X_train, y_train, X_dev, y_dev
    
    
    
# =================== #
# Data pre-processing #

numcep = 40     # Mel-Frequency Cepstrum Coefficients, default 12.
numfilt = 40    # Number of filters in the filterbank, default 26.

winlen = 0.025  # Length of the analysis window in seconds. Default is 0.025s (25 milliseconds).
winstep = 0.01  # Step between successive windows in seconds. Default is 0.01s (10 milliseconds).
method = 1      # If 1: compute derivative up to 3rd order. If 0, derivative up to 2nd order.\
                # Default is 1

param_set = "lmfb" + str(numcep) + "-" + str(numfilt) + "win" + "-" + str(int(winlen*1000)) + "-" +str(int(winstep*1000))


# ============================== #
# Get speakers ids (directories) # 

train_speaker, valid_speaker = get_speaker_lists(os.path.join(ROOT_DATA_DIR,"TRAIN/"))
print("train_set speakers: ", train_speaker.__len__())
print("validation_set speakers: ", valid_speaker.__len__())


# ========================== #
# Create phonemes dictionary # 

dic_location = "preprocessing_TIMIT/phonemlist"
timit_dict = get_timit_dict(dic_location)



# =================================== #
# Create training and validation sets # 

train_set_x = []
train_set_y = []
valid_set_x = []
valid_set_y = []

size_dict = {}

print("Loading train and validation sets:\n")
dirlist = ['DR5', 'DR6', 'DR7', 'DR3', 'DR2', 'DR1', 'DR4', 'DR8']
for d in dirlist:
    # print("\nDirectory: ", d)
    path = os.path.join(ROOT_DATA_DIR,os.path.join("TRAIN",d))
    # print("Data dir: ",d)
    
    for dirName, subdirList, fileList in tqdm( os.walk(path) , ncols=100,desc=d):
        # [check] # print('Found directory: %s' % dirName)
        
        path,folder_name = os.path.split(dirName)
        # [check] # print('Speaker: ' + folder_name)
        if folder_name.__len__() >= 1:
            temp_name = ""
            for fname in sorted(fileList):
                name = fname.split(".")[0]
                if name != temp_name:
                    temp_name = name
                    # [check] # print('\t%s' % dirName+"/"+name)
                    #wav_location = dirName+"/"+name+".WAV"
                    wav_location = dirName+"/"+name+".wav"
                    phn_location = dirName+"/"+name+".PHN"
                    
                    feat, sample, rate = get_features(wav_location, numcep, numfilt, winlen, winstep, method,quaternion=True)
                    
                    # [check] # print(feat.shape)
                    input_size = feat.shape[0]
                    target = get_target(phn_location,timit_dict, input_size)
                    if folder_name in train_speaker:
                        train_set_x.append(feat)
                        train_set_y.append(target)
                    elif folder_name in valid_speaker:
                        valid_set_x.append(feat)
                        valid_set_y.append(target)
                    else:
                        assert False, "unknown name"
    # save_data_to_tensor(train_set_x, train_set_y, d, 'train_', param_set)
    # save_data_to_tensor(valid_set_x, valid_set_y, d, 'val_', param_set)
    # train_set_x = []
    # train_set_y = []
    # valid_set_x = []
    # valid_set_y = []
save_data_to_tensor(train_set_x, train_set_y, 'train_', param_set)
save_data_to_tensor(valid_set_x, valid_set_y, 'val_', param_set)


##################
# Save .npz file #
train_file_name = datasetDir + "/timit_" + "train_" + "xy_" + param_set + ".npz"
valid_file_name = datasetDir + "/timit_" + "val_" + "xy_" + param_set + ".npz"

np.savez(train_file_name, X_train_t, y_train_t)
np.savez(valid_file_name, X_dev_t, y_dev_t)
#####

train_data_name = datasetDir + "/timit_" + "train-val_" + "xy_" + param_set + ".npz"
np.savez(train_data_name, x_train=X_train_t, y_train=y_train_t, x_val=X_dev_t, y_val=y_dev_t)
#####

npzfile_train = np.load(train_data_name)

#####

with np.load(train_data_name) as data:
    X_train_t = data['x_train']
    y_train_t = data['y_train']
    X_dev_t = data['x_val']
    y_dev_t = data['y_val']
    
###################
print("write valid set")
print("valid set length: " + str(valid_set_x.__len__()))


file_name = datasetDir + "timit_" + "valid_" + "xy_" + para_name + ".klepto"
# print("valid set name: " + file_name)
# d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
# d['x'] = valid_set_x
# d['y'] = valid_set_y
# d.dump()
# d.clear()

# print("write train set")
# print("train set length: " + str(train_set_x.__len__()))
# file_name = datasetDir + "timit_" + "train_" + "xy_" + para_name + ".klepto"
# print("train set name: " + file_name)
# d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
# d['x'] = train_set_x
# d['y'] = train_set_y
# d.dump()
# d.clear()


# test_set_x = []
# test_set_y = []

# for d in dirlist:
#     path = os.path.join(ROOT_DATA_DIR,os.path.join("TEST",d))
#     for dirName, subdirList, fileList in os.walk(path):
#         print('Found directory: %s' % dirName)

#         path,folder_name = os.path.split(dirName)
#         print('Speaker: ' + folder_name)
#         if folder_name.__len__() >= 1:
#             temp_name = ""
#             for fname in sorted(fileList):
#                 name = fname.split(".")[0]
#                 if name != temp_name:
#                     temp_name = name
#                     print('\t%s' % dirName+"/"+name)
#                     wav_location = dirName+"/"+name+".wav"
#                     phn_location = dirName+"/"+name+".PHN"
#                     feat, _, _ = get_features(wav_location, numcep, numfilt, winlen, winstep, method, quaternion=True)
#                     print(feat.shape)
#                     input_size = feat.shape[0]
#                     target = get_target(phn_location,timit_dict, input_size)
#                     test_set_x.append(feat)
#                     test_set_y.append(target)



# print("write test set")
# print("test set length: " + str(test_set_x.__len__()))
# file_name = datasetDir + "timit_" + "test_" + "xy_" + para_name + ".klepto"
# print("test set name: " + file_name)
# d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
# d['x'] = test_set_x
# d['y'] = test_set_y
# d.dump()
# d.clear()

"""
size_dict = get_data_maxlen(ROOT_DATA_DIR)

max(size_dict.keys())
min(size_dict.keys())
size_dict[778]
size_dict[91]
"""