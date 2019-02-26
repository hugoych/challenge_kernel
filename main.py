from preprocessing import convert_data
from svm import my_svm, spectrum_kernel,linear_kernel, gaussian_kernel
import pandas as pd 
from sklearn.metrics import accuracy_score
import numpy as np


print(accuracy_score([0,0],[1,1]))

data_train = pd.read_csv('data/Xtr0.csv', index_col=0)
data_train2 = pd.read_csv('data/Xtr1.csv', index_col=0)
data_train3 = pd.read_csv('data/Xtr2.csv', index_col=0)
data_train = pd.concat([data_train,data_train2,data_train3])
data_train = np.array([convert_data(x) for x in data_train['seq']])
data_train = data_train[:-600]

label_train = pd.read_csv('data/Ytr0.csv', index_col=0)
label_train2 = pd.read_csv('data/Ytr1.csv', index_col=0)
label_train3 = pd.read_csv('data/Ytr2.csv', index_col=0)
label_train = pd.concat([label_train,label_train2,label_train3])
label_train = label_train['Bound'].values
label_train = np.array([2*x-1 for x in label_train])
label_train = label_train[:-600]

data_test = pd.read_csv('data/Xtr1.csv', index_col=0)
data_test = np.array([convert_data(x) for x in data_test['seq']])
data_test = data_test[-600:]

label_test = pd.read_csv('data/Ytr1.csv', index_col=0)
label_test = label_test['Bound'].values
label_test = np.array([2*x-1 for x in label_test])
label_test = label_test[-600:]

svm_spectrum = my_svm(kernel=spectrum_kernel,C=1.)

K_spect = svm_spectrum.fit(data_train,label_train)

svm_linear = my_svm(kernel=gaussian_kernel,C=1.)

K_lin = svm_linear.fit(data_train,label_train)


pred1 = svm_spectrum.predict(data_test)

print('accuracy spectrum',accuracy_score(label_test,pred1))


pred2 = svm_linear.predict(data_test)

print('accuracy linear', accuracy_score(label_test,pred2))

print(label_test)
print(pred1)
print(pred2)
print(pred1-pred2)