import numpy as np
import pandas as pd
from preprocessing import convert_data


def load_training_data(i):
    data = pd.read_csv('data/Xtr%i.csv' % i, index_col=0)
    data = np.array([convert_data(x) for x in data['seq']])
    label = pd.read_csv('data/Ytr%i.csv' % i, index_col=0)
    label = 2 * label['Bound'].values - 1
    return data, label

def load_test_data(i):
    data = pd.read_csv('data/Xte%i.csv' % i, index_col=0)
    data = np.array([convert_data(x) for x in data['seq']])
    return data


def load_mat_data():
    data = [pd.read_csv('data/Xtr%i_mat100.csv' % i, sep=' ', header=None) for i in range(3)]
    label = [pd.read_csv('data/Ytr%i.csv' % i, index_col=0) for i in range(3)]
    data = pd.concat(data).values
    label = 2 * pd.concat(label)['Bound'].values - 1
    return data, label


def split(data, label, training, validation):
    assert training + validation == 1
    n = len(data)
    perm = np.random.permutation(n)
    data = data[perm]
    training_index = int(training * n)
    training = (data[:training_index], label[:training_index])
    validation = (data[training_index:], label[training_index:])
    return training, validation

