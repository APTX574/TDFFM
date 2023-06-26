import numpy as np
import pandas as pd
import torch

from sklearn.datasets._base import Bunch
from sklearn import preprocessing

abs = "./"


def change(x):
    x = x.reshape((x.shape[0], 5, x.shape[1] // 5))
    x = x.transpose(2, 1)
    return x


def load(coding_types):
    train = Bunch()
    test = Bunch()
    train.data = {}
    test.data = {}
    for coding_type in coding_types:
        if coding_type != "one_hot":
            print(coding_type)
            data_train_csv = pd.read_csv(abs + "dataset/data/train_set_{}.csv".format(coding_type), header=None)
            data_test_csv = pd.read_csv(abs + "dataset/data/test_set_{}.csv".format(coding_type), header=None)
            train.data[coding_type] = _get_3cl_data(data_train_csv)
            test.data[coding_type] = _get_3cl_data(data_test_csv)
    data_test_one_hot = pd.read_csv(abs + "dataset/data/test_set.csv", header=None)
    data_train_one_hot = pd.read_csv(abs + "dataset/data/train_set.csv", header=None)
    test.target = _get_3cl_target(data_test_one_hot)
    train.target = _get_3cl_target(data_train_one_hot)
    if "one_hot" in coding_types:
        data_onehot_train = np.array(data_train_one_hot[:])[:, :-1]
        data_onehot_test = np.array(data_test_one_hot[:])[:, :-1]
        enc = preprocessing.OneHotEncoder()
        enc.fit(data_onehot_train)
        one_hot_train = torch.Tensor(enc.transform(data_onehot_train).toarray())
        one_hot_test = torch.Tensor(enc.transform(data_onehot_test).toarray())
        test.data["one_hot"] = one_hot_test
        train.data["one_hot"] = one_hot_train

    return train, test


def _get_3cl_data(data):
    data_r = data.iloc[:, 1:]
    data = np.array(data_r, dtype=np.float)
    print(data.shape)
    return torch.Tensor(data)


def _get_3cl_target(data):
    data_b = data.iloc[:, -1:]
    data_np = np.array(data_b, dtype=np.long).flatten()
    return torch.Tensor(data_np).type(torch.LongTensor)


def get_index(key):
    index_csv = pd.read_csv(abs + "dataset/data/aaindex.csv")
    index = index_csv["AA"].tolist()
    i = index.index(key)
    return np.array(index_csv.iloc[i:i + 1]).flatten()[1:]
