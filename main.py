# -*- coding: utf-8 -*-

import preprocessing
import models
import training

from preprocessing  import get_data
from models import DNN_model, QNN_model
from training import train, model_plots

def main():
    print("-------------------\n")
    print("Load training data:\n")
    train_data, val_data, n_classes = get_data()
    
    print("\n\n-------------\n")
    print("Create model:\n")
    model = DNN_model(n_classes)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    print("\n\n-----------\n")
    print("Train model:\n")
    history = train(model, train_data, val_data)

    print("\n\n------------------\n")
    print("Accuracy / loss plots:\n")
    model_plots(history)

if __name__ == "__main__":
    main()
    