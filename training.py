# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


""" Performs training of network model on training and validation sets
params
    model: NN model to train
    train_data: training features and lables set
    val_data: validation features and lables set
return
    history: training history
"""
def train(model, train_data, val_data):
    X_train, y_train = train_data
    X_dev, y_dev = val_data
    
    history = model.fit(x=X_train, 
                        y=y_train,
                        epochs=20, 
                        #callbacks=[early_stop, checkpoint], 
                        batch_size=32, 
                        validation_data=(X_dev, y_dev)
                        )
    
    return history


""" Plots accuracy and loss of training
params
    history: training history
return
    None
"""
def model_plots(history):
    h = history.history

    # summarize history for accuracy
    plt.plot(h['accuracy'])
    plt.plot(h['val_accuracy'])
    # plt.title('Training')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    ax = plt.axes()        
    ax.yaxis.grid()
    plt.show()
    
    # summarize history for loss
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    # plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    ax = plt.axes()        
    ax.yaxis.grid()
    plt.show()