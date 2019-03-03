import numpy as np
import pandas as pd
import utils
import kernel_methods

data, label = utils.load_training_data()
# data = (data - data.mean(0))/data.std(0)
training, validation = utils.split(data, label, 0.8, 0.2)
data.mean()
svm = kernel_methods.SVM('spectrum', C=3.0, kernel_param={'p': 5, 'var': 2.})
svm.X, svm.Y = training
svm.K = svm.build_K()
svm.solve()
print(svm.score(*training))
print(svm.score(*validation))
