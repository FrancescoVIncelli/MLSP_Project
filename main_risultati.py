# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 08:55:21 2020

@author: -
"""

from asr_models_2 import sequential_cnn_1D, simple_cnn_1D, simple_cnn_2D
import tensorflow as tf
import numpy as np
from dataclasses import dataclass
import matplotllib.pyplot as plt

def pad_labels(label_batch, max_str_len, categ=False, pad_value=60):
    batch_padded = []
    print(pad_value)
    for label in label_batch:
        pad_len = max_str_len-len(label)
        if pad_len == 0:
            if categ:
                label_padded = tf.keras.utils.to_categorical(label, num_classes=61, dtype='int32')
            else:
                label_padded = label
                
        else:
            if categ:
                label_padded = np.pad(label, ((0,pad_len)), constant_values=pad_value)
                label_padded = tf.keras.utils.to_categorical(label_padded, num_classes=61, dtype='int32')
            else:
                label_padded = np.pad(label, ((0,pad_len)), constant_values=pad_value)
        batch_padded.append(label_padded)
            
    return np.array(batch_padded, dtype='int32')


def pad_feats(feat_bacth, max_time_step, pad_value=None, pad_mode='last'):    
    batch_padded = []
    
    if pad_mode=='last':
        for feat in feat_bacth:
            pad_step = max_time_step-feat.shape[0]
            
            if pad_step == 0:
                batch_padded.append(feat)
            else:
                if pad_value is None:
                    last = feat[-1]
                else:
                    last = np.array([pad_value]*160, dtype='float64')
                    
                pad = np.array([last for i in range(pad_step)],dtype='float64')
                #print(feat.shape, pad.shape)
                feat_padded = np.append(feat,pad,0)
                batch_padded.append(feat_padded)
        
    return np.array(batch_padded,dtype='float32')


def trainRunner(model, train_tensors, valid_tensors, args, verbose=1):
    
    trainX, trainY = train_tensors
    validX, validY = valid_tensors
    
    # ==== Run training ==== #
    batch_size = args.batch_size
    epochs_num = args.epochs_num
    save_path  = args.save_path
    
    print("Training report:")
    print("batch_size: ", batch_size)
    print("epochs_num: ", epochs_num)
    # print("trainX: ", trainX)
    # print("\nvalidX: ", validX)
    # print("\ntrainY: ", trainY)
    # print("\nvalidY: ", validY)
    
    try:
        history = model.fit(
            x=trainX,
            y=trainY,
            batch_size=batch_size,
            epochs=epochs_num,
            validation_data=( validX, validY ),
            shuffle=True,
            verbose=1
        )
    except KeyboardInterrupt:
        # serialize model to JSON
        #### model_json = model.to_json()
        #### with open("model.json", "w") as json_file:
        ####     json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(train_args.save_path)
        print("Saved model to disk")
        return None

    return history

@dataclass
class ModelArgs:
    model_class:str
    model_type:str
    input_shape:list
    n_layers_convs:int
    n_layers_dense:int
    kernel_size:tuple
    num_classes:int
    act:str
    aact:str
    padding:str
    l2_reg:float
    shared_axes:list
    opt:str
    kernel_init:str='glorot_normal'
    max_str_len:int = None
    sf_dim:int=8
    dropout_prob:float=0.3
    

@dataclass
class TrainArgs:
    network_args:type
    lr_method:str
    epochs_num:int
    batch_size:int
    save_path:str
    


model_args = ModelArgs(
    model_class="simple_cnn_2D",
    model_type='real',
    input_shape=[778,4,40],
    n_layers_convs=4,
    n_layers_dense=3,
    sf_dim=8,
    kernel_size=(3,3),
    num_classes=61,
    act='linear',
    aact='prelu',
    padding='same',
    l2_reg=1e-5,
    shared_axes=[0,1],
    opt='adam',
    kernel_init='glorot_normal',
    max_str_len=75
)

train_args = TrainArgs(
    network_args = model_args,
    lr_method="ctc",
    epochs_num=40,
    batch_size=4,
    save_path="Simple-CNN-2D_TYP-real_LV-phn_NE-20_BS-4_DATAPREP-TIMITspeech_1stTrainSess.h5"
)


# ===========================
# Loading and preprocess data

# train_data_path="./data_set/timit_train_Xy_methodTS_PHN.pickle"
train_data_path="./data_set/timit_train_Xy_methodTS_PHN_CTC.pickle"   
train_dataset = load_pickle(train_data_path)

# valid_data_path="./data_set/timit_valid_Xy_methodTS_PHN.pickle"
valid_data_path="./data_set/timit_valid_Xy_methodTS_PHN_CTC.pickle"   
valid_dataset = load_pickle(valid_data_path)

# ==== Split train and valid datasets ==== #

# Train data
X_train = train_dataset['x_train']
y_train = train_dataset['y_train']
# Validation data
X_valid = valid_dataset['x_valid']
y_valid = valid_dataset['y_valid']

# ===========================
# Get model tensors (1D-Conv)

train_data = (X_train, y_train)
valid_data = (X_valid, y_valid)
train_tensors, valid_tensors = modelTensorsPreparation(train_dataset = train_data,
                                                       valid_dataset = valid_data,
                                                       learning_method='ctc')

# Needed for 2D-Convolution
trainX = train_tensors[0]['the_input']
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 4, 40))
train_tensors[0]['the_input'] = trainX

validX = valid_tensors[0]['the_input']
validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 4, 40))
valid_tensors[0]['the_input'] = validX

# =============
# Check tensors (OK)

inputs = valid_tensors[0]
inputs['the_input'].shape

# ============
# Create model
model = simple_cnn_2D(model_args) # simple_cnn_1D(model_args) # sequential_cnn_1D(model_args)

# =================================
# Categoricla (sequential) training
tensor_train = (train_tensors[0]['the_input'], train_tensors[1]['the_output'])
tensor_valid = (valid_tensors[0]['the_input'], valid_tensors[1]['the_output'])

train_history = trainRunner(model, tensor_train, tensor_valid, train_args)

# =================
# CTC-Loss training
train_history_2D_2nd = trainRunner(model, train_tensors, valid_tensors, train_args)

# Save model weights
model.save_weights(train_args.save_path)
print("Saved model to disk")
        
# ================
# Plot loss metric

if train_history_2D_2nd:
    hist = train_history_2D_2nd.history
      
    # summarize history for loss
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    # plt.title('Loss')
    plt.ylabel('CTC-Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    ax = plt.axes()        
    ax.yaxis.grid()
    plt.show()
