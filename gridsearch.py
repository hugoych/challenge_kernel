# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:55:35 2019

@author: hugol
"""
import numpy as np
import pandas as pd
import utils
import kernel_methods
from time import time
from sklearn.model_selection import train_test_split


kernels = [['mismatch', {'var':2.,'k' : 8 }],['mismatch', {'k' : 4 }],['mismatch', {'k' : 5 }],['mismatch', {'k' : 6 }],['mismatch', {'k' : 7 }]]
Cs = [0.1,0.5,1.0,2.0,5.0]

GridSearch  = kernel_methods.Hyper(kernels,Cs)

for i in range(3):
    data, label = utils.load_training_data(i)
    data_train, data_val, y_train, y_val = train_test_split(data, label, stratify=label, train_size=0.8)
    data_test = utils.load_test_data(i)
    training = data_train, y_train
    validation = data_val, y_val
    GridSearch.boost(training,validation)
    