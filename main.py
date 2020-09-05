# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:01:37 2020

@author: vince
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from   tensorflow.keras.optimizers          import SGD, Adam, RMSprop
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Lambda, Input
import functools

tf.random.set_seed(19)
print("TensorFlow version: ",tf.__version__)

# Load the TensorBoard notebook extension
# %load_ext tensorboard
# tensorboard --logdir logs

# from interspeech_model import interspeech_TIMIT_Model
from ctc_interspeech_model import QCNN_model # ctc_interspeech_TIMIT_Model, ctc_loss, QCNN_model

# from module import DS2_timit

""" TIMIT Conv.Only model args """
   
parser = argparse.ArgumentParser(description='')
parser.add_argument('--num_layers', dest='num_layers', type=int, default=6, help='# of layers of the network model')
parser.add_argument('--start_filter', dest='start_filter', type=int, default=8, help='filter size of the 1st Conv Layer')
parser.add_argument('--act', dest='act', default='relu', help='layers activation functions')
parser.add_argument('--aact', dest='aact', default='relu', help='layers advanced activation functions')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.3, help='dropout probability')
parser.add_argument('--l2', dest='l2', type=float, default=1e-5, help='L2 regularization')
parser.add_argument('--model', dest='model', default='quaternion', help='model version: real / quaternion')
parser.add_argument('--quat_init', dest='quat_init', default='quaternion', help='kernel initializers for quaternion model')
parser.add_argument('--ctc', dest='ctc', type=bool, default=False, help='use ctc loss')
args = parser.parse_args()
# parser.add_argument('--dataset_dir', dest='dataset_dir', default='city', help='path of the dataset')
# parser.add_argument('--epoch', dest='epoch', type=int, default=1, help='# of epoch')
# parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=1, help='# of epoch to decay lr')
# parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
# parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
# parser.add_argument('--phase', dest='phase', default='train', help='train, test')

# parser.add_argument('--num_classes', dest='num_classes', type=int, default=34, help='number of segmentation classes')
# parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
# parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
# parser.add_argument('--print_freq', dest='print_freq', type=int, default=5, help='print the debug information every print_freq iterations')
# parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
# parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
# parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
# parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')


# q_model, val_function = getTimitModel2D(args)
q_model = interspeech_TIMIT_Model(args)
q_model.summary()


#
# Load Train and Validation Data
#
data_path = "./data_set/timit_train-val_xy_lmfb40-40win-25-10.npz"
dataset = np.load(data_path)


# X_train_t = dataset['x_train']
# y_train_t = dataset['y_train']
X_dev_t   = dataset['x_val']
y_dev_t   = dataset['y_val']


model_output = q_model(X_dev_t)

#########################
#### Custom Training ####

BATCH_SIZE = 10

def pack_features_vector(features, labels):
    return features, labels

## Create dataset iterator
dataset_loader = tf.data.Dataset.from_tensor_slices((X_dev_t, y_dev_t)).batch(BATCH_SIZE)
# (not used) dataiter_map = dataset_loader.map(pack_features_vector)

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

def ctc_loss(y_true, y_pred):
    """
    input_length = np.array(([61]*y_true.shape[1])).reshape((1,-1))
    label_length = np.array(([61]*y_pred.shape[1])).reshape((1,-1))
    
    input_length = tf.convert_to_tensor(input_length, dtype='int64')
    label_length = tf.convert_to_tensor(label_length, dtype='int64')
    """
    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

labels = Input(name='the_labels', shape=[None], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
pred = Input(name='y_pred', shape=[None], dtype='float32') 
   
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_object = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred, labels,input_length,label_length])

def loss(model, X, y, training):
    """
    print("[*] loss()")
    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)
    """
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    pred_y = model(X, training=training)
    #### print("y_pred.shape: ", pred_y.shape)
    
    object_loss = loss_object(y_true=y, y_pred=pred_y)
    #### print("loss_object CALLED")
    
    return object_loss

#######################
#### Training Loop ####

#### Gradient Descending ####
def grad(model, inputs, targets):
    #### print("[*] grad()")
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

#### Optimizer ####
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 10

for epoch in range(num_epochs):
    print("[****] Epoch {:03d}".format(epoch))
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    # Training loop - using batches of 10
    for X, y in dataset_loader:
        # Optimize the model
        loss_value, grads = grad(q_model, X, y)
        optimizer.apply_gradients(zip(grads, q_model.trainable_variables))
      
        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(y, q_model(X, training=True))
    
    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    
    #if epoch % 50 == 0:
    print(" [*] Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))



# =====================
# Keras 'fit'Training #
# =====================
    
# Train Data
# x_train = np.array(train_set_x).astype(np.float32)
# y_train = np.array(train_set_y).astype(np.float32)
# Validation Data
# x_dev = np.array(valid_set_x).astype(np.float32)
# y_dev = np.array(valid_set_y).astype(np.float32)

#
# Compute network parameters
#
nb_labels = 61 # number of labels (10, this is digits)
batch_size = 10 # size of the batch that are considered
nb_train = len(X_dev_t)
nb_features = len(X_dev_t[0])


# create list of input lengths
x_train_len = np.asarray([len(X_dev_t[i]) for i in range(nb_train)])
y_train_len = np.asarray([len(y_dev_t[i]) for i in range(nb_train)])

opt = Adam(lr = 0.001)
q_model.compile(optimizer=opt)

"""
history = q_model.fit(X_dev_t, y_dev_t,
                      # validation_data=(X_dev_t,y_dev_t),
                      epochs=4,
                      batch_size=10)
"""
# CTC training
history = q_model.fit(x=[X_dev_t, y_dev_t, x_train_len, y_train_len], y=y_dev_t, #y=np.zeros((nb_train, nb_features, nb_labels)),
                      batch_size=batch_size, epochs=4)
#
# Eval.
#
# loss, acc = q_model.evaluate(x_test,y_test)
# print('Test Loss = '+str(loss)+' | Test accuracy = '+str(acc))
# print("That's All Folks :p ")


# ======================= #
# Use CTC Loss (Training) #
# ======================= #
# [FAST_SEARCH] #
# =====================
# Load Training Dataset
data_path = "./data_set/timit_train-val_xy_lmfb40-40win-25-10.npz"
dataset = np.load(data_path)

# == Split train and validation data ==
# X_train_t = dataset['x_train']
# y_train_t = dataset['y_train']
X_dev_t   = dataset['x_val']
y_dev_t   = dataset['y_val']


# ======================
# Load and compile model

# q_model, label_length_ts, pred_length_ts, y_true_input_ts = ctc_interspeech_TIMIT_Model(args)
q_model, conv_shape = QCNN_model() # ctc_interspeech_TIMIT_Model(args)
q_model.summary()
# [deleted] ctc_loss_prepare_fn = functools.partial(ctc_loss, input_length=pred_length_ts, label_length=label_length_ts, real_y_true_ts=y_true_input_ts)
# [deleted] q_model.compile(optimizer=Adam(lr=0.0001), loss=ctc_loss_prepare_fn)



# ==================
# Train model inputs
n_len = 61

trainNum = X_dev_t.shape[0];
# (not used) # validNum = valid_X.shape[0];

trainLabel = np.zeros((trainNum, n_len), dtype=np.uint8)
# (not used) # validLabel = ...

trainInputL = np.ones(trainNum)*int(conv_shape[1]-2)
trainLabelL = np.ones(trainNum)*n_len
# (not used) # validInputL = np.ones(validNum)*int(conv_shape[1]-2)
# (not used) # validLabelL = np.ones(validNum)*n_len


# ===========
# Train model

history = q_model.fit([X_dev_t, trainLabel, trainInputL, trainLabelL],
                      trainLabel,
                      batch_size=10,
                      epochs=2,
                      shuffle=True,
                      verbose=1)
                      #validation_data=([validX,validLabel,validInputL,validLabelL], validLabel))

#model.save('plateCTC.h5')
q_model.save('q_model_timit_ctc.h5')


# (modified)
# history = q_model.fit(x=[X_dev_t, pred_length_ts, label_length_ts, y_true_input_ts],
#                       y=y_dev_t, #y=np.zeros((nb_train, nb_features, nb_labels)),
#                       batch_size=batch_size, epochs=4)



"""
DS2_parser = argparse.ArgumentParser(description='')
DS2_parser.add_argument('--num_conv_layers', dest='num_conv_layers', type=int, default=4, help='# of convolutional layers')
DS2_parser.add_argument('--num_recur_layers', dest='num_recur_layers', type=int, default=6, help='# of recurrent layers')
DS2_parser.add_argument('--start_filter', dest='start_filter', type=int, default=32, help='filter size of the 1st Conv Layer')
DS2_parser.add_argument('--act', dest='act', default='relu', help='layers activation functions')
DS2_parser.add_argument('--aact', dest='aact', default='prelu', help='layers advanced activation functions')
DS2_parser.add_argument('--dropout', dest='dropout', type=float, default=0.3, help='dropout probability')
DS2_parser.add_argument('--l2', dest='l2', type=float, default=1e-5, help='L2 regularization')
DS2_parser.add_argument('--model', dest='model', default='real', help='model version: real / quaternion')
DS2_parser.add_argument('--quat_init', dest='quat_init', default='quaternion', help='kernel initializers for quaternion model')
DS2_parser.add_argument('--use_ctc', dest='use_ctc', type=bool, default=False, help='use ctc loss')
DS2_args = DS2_parser.parse_args()

DS2_real = DS2_timit(DS2_args)
"""

import sys
try:
    while(True):
        a=1
except:
    e = sys.exc_info()[0]
    print( "Error: %s" %(e) )