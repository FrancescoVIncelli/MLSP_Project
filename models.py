# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Flatten, Input, BatchNormalization ,\
    GRU, Bidirectional, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model

#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

import quaternion_nn
from quaternion_nn import QuaternionConv1D, QuaternionDense
#K.clear_session()


""" Create a non-quaternion DNN model
prams
    n_classes: number of classes for classification task
return
    model:CRNN model
"""
def DNN_model(n_classes):
    inputs = Input(shape=(98,40))
    batch_norm = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(inputs)
    
    # First Conv1D layer
    conv1 = Conv1D(32,2, padding='valid', activation='relu', strides=1)(batch_norm)
    pool1 = MaxPooling1D(pool_size=3)(conv1) # MaxPooling1D(pool_size=(3,3))(conv1)
    d1 = Dropout(0.3)(pool1)
    
    # Second Conv1D layer
    conv2 = Conv1D(64, 2, padding='valid', activation='relu', strides=1)(d1)
    pool2 = MaxPooling1D(pool_size=3)(conv2) # MaxPooling2D(pool_size=(3,3))(conv2)
    d2 = Dropout(0.3)(pool2)
    
    # Third Conv1D layer
    conv3 = Conv1D(128, 2, padding='valid', activation='relu', strides=1)(d2)
    pool3 = MaxPooling1D(pool_size=3)(conv3) # MaxPooling2D(pool_size=(3,3))(conv3)
    d3 = Dropout(0.3)(pool3)
    
    # RNN
    batch_norm2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(d3)
    bgru1 = Bidirectional(GRU(units=128, return_sequences=True), merge_mode='sum')(batch_norm2)
    bgru2 = Bidirectional(GRU(units=128, return_sequences=True), merge_mode='sum')(bgru1)
    bgru3 = Bidirectional(GRU(units=128, return_sequences=False), merge_mode='sum')(bgru2)
    batch_norm3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(bgru3)
    
    # Flatten layer
    flat = Flatten()(batch_norm3)
    
    # Dense Layer 1
    dense = Dense(256, activation='relu')(flat)
    # Outputs (Dense layer 2)
    outputs = Dense(n_classes, activation="softmax")(dense)
    
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    
    return model


""" Create a Quaternion CNN model
prams
    n_classes: number of classes for classification task
return
    model: QUaternion-CNN model
"""
def QNN_model(n_classes):
    inputs = Input(shape=(98,40))
    
    # First QConv1D layer
    conv1  = QuaternionConv1D(32, 2, strides=1, activation="relu", padding="valid")(inputs)
    # print("conv1:\n",conv1)
    pool1  = MaxPooling1D(pool_size=2)(conv1)
    # print("pool1:\n",pool1)
    
    # Second QConv1D layer
    conv2  = QuaternionConv1D(64, 2, strides=1, activation="relu", padding="valid")(pool1)
    # print("conv2:\n", conv2)
    pool2  = MaxPooling1D(pool_size=2)(conv2)
    # print("pool2:\n", pool2)
    
    # Third QConv1D layer
    conv3  = QuaternionConv1D(128, 2, strides=1, activation="relu", padding="valid")(pool2)
    # print("conv3:\n", conv3)
    pool3  = MaxPooling1D(pool_size=2)(conv3)
    # print("pool3:\n", pool3)
    
    # FLatten layer
    flat   = Flatten()(pool3)
    # print("flat:\n", flat)
    
    # Dense layer 1
    dense  = QuaternionDense(256, activation='relu')(flat)
    
    # Dense layer 2
    # dense  = QuaternionDense(64, activation='relu')(dense)
    # print("dense:\n", dense)
    
    outputs = Dense(n_classes, activation='softmax')(dense)
    
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    
    return model