import numpy as np
import pandas as pd
import utils
import kernel_methods
from time import time
from sklearn.model_selection import train_test_split

predictions = []

for i in range(3):
    data, label = utils.load_training_data(i)
    data_train, data_val, y_train, y_val = train_test_split(data, label, train_size=0.8)
    data_test = utils.load_test_data(i)
    training = data_train, y_train
    validation = data_val, y_val

    svm = kernel_methods.SVM('spectrum', C=1.0, kernel_param={'p': 5, 'var': 2.})
    svm.X, svm.Y = data, label
    svm.K = svm.build_K()
    svm.solve()
    print(svm.score(*training))
    predictions.append(svm.predict(data_test))

