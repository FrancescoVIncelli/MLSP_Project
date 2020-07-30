# -*- coding: utf-8 -*-

# =============== #
# Import packages #

from __future__ import print_function
import os
import klepto                                                         # Save pre-processed data
from utility import print_line                                        # clearer stdout printing funtion

from preprocessing_TIMIT.targets import get_target, get_timit_dict    # Create target variables

from preprocessing_TIMIT.features import get_features                 # Extract input features
                                                                      # for model training

from preprocessing_TIMIT.speakers import get_speaker_lists            # (TODO)


# Set the path to the root directory containing the TIMIT dataset
ROOT_DATA_DIR = "./data/lisa/data/timit/raw/TIMIT"

# Check that TRAIN and TEST data folder are not empty
if "TEST/" in os.listdir(ROOT_DATA_DIR):
    raise Warning("TIMIT TEST data is missing")
if "TRAIN/" in os.listdir(ROOT_DATA_DIR):
    raise Warning("TIMIT TRAIN data is missing")

# Location of the target data set folder
datasetDir = "data_set/"


# =================== #
# Data pre-processing #

numcep = 40     # Mel-Frequency Cepstrum Coefficients, default 12.
numfilt = 40    # Number of filters in the filterbank, default 26.

winlen = 0.025  # Length of the analysis window in seconds. Default is 0.025s (25 milliseconds).
winstep = 0.01  # Step between successive windows in seconds. Default is 0.01s (10 milliseconds).
method = 1      # If 1: compute derivative up to 3rd order. If 0, derivative up to 2nd order.\
                # Default is 1

para_name = "lmfb" + str(numcep) + "-" + str(numfilt) + "win" + "-" + str(int(winlen*1000)) + "-" +str(int(winstep*1000))


# ============================== #
# Get speakers ids (directories) # 

train_speaker, valid_speaker = get_speaker_lists(os.path.join(ROOT_DATA_DIR,"TRAIN/"))
print_line("train_set speakers: ", train_speaker.__len__())
print_line("validatio_set speakers: ", valid_speaker.__len__())


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

dirlist = ['DR5', 'DR6', 'DR7', 'DR3', 'DR2', 'DR1', 'DR4', 'DR8']
for d in dirlist:
    path = os.path.join(ROOT_DATA_DIR,os.path.join("TRAIN",d))
    for dirName, subdirList, fileList in os.walk(path):
        print('Found directory: %s' % dirName)

        path,folder_name = os.path.split(dirName)
        print('Speaker: ' + folder_name)
        if folder_name.__len__() >= 1:
            temp_name = ""
            for fname in sorted(fileList):
                name = fname.split(".")[0]
                if name != temp_name:
                    temp_name = name
                    print('\t%s' % dirName+"/"+name)
                    #wav_location = dirName+"/"+name+".WAV"
                    wav_location = dirName+"/"+name+".wav"
                    phn_location = dirName+"/"+name+".PHN"
                    feat, _, _ = get_features(wav_location, numcep, numfilt, winlen, winstep, method,quaternion=True)
                    print(feat.shape)
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



print("write valid set")
print("valid set length: " + str(valid_set_x.__len__()))
file_name = datasetDir + "timit_" + "valid_" + "xy_" + para_name + ".klepto"
print("valid set name: " + file_name)
d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
d['x'] = valid_set_x
d['y'] = valid_set_y
d.dump()
d.clear()

print("write train set")
print("train set length: " + str(train_set_x.__len__()))
file_name = datasetDir + "timit_" + "train_" + "xy_" + para_name + ".klepto"
print("train set name: " + file_name)
d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
d['x'] = train_set_x
d['y'] = train_set_y
d.dump()
d.clear()


test_set_x = []
test_set_y = []

for d in dirlist:
    path = os.path.join(ROOT_DATA_DIR,os.path.join("TEST",d))
    for dirName, subdirList, fileList in os.walk(path):
        print('Found directory: %s' % dirName)

        path,folder_name = os.path.split(dirName)
        print('Speaker: ' + folder_name)
        if folder_name.__len__() >= 1:
            temp_name = ""
            for fname in sorted(fileList):
                name = fname.split(".")[0]
                if name != temp_name:
                    temp_name = name
                    print('\t%s' % dirName+"/"+name)
                    wav_location = dirName+"/"+name+".wav"
                    phn_location = dirName+"/"+name+".PHN"
                    feat, _, _ = get_features(wav_location, numcep, numfilt, winlen, winstep, method, quaternion=True)
                    print(feat.shape)
                    input_size = feat.shape[0]
                    target = get_target(phn_location,timit_dict, input_size)
                    test_set_x.append(feat)
                    test_set_y.append(target)



print("write test set")
print("test set length: " + str(test_set_x.__len__()))
file_name = datasetDir + "timit_" + "test_" + "xy_" + para_name + ".klepto"
print("test set name: " + file_name)
d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
d['x'] = test_set_x
d['y'] = test_set_y
d.dump()
d.clear()