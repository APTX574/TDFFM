import os
import re

import joblib
import torch
import xgboost as xgb
from deepforest import CascadeForestClassifier
from sklearn.ensemble import *

from model.ablation import tf_d_4
from model.deep1 import deep1
from model.lstm import lstm1


def get_net(net_name, seq_len, channel, coding_type, new=True, _ep=0):
    if net_name == "RF":
        if not new:
            pth = "./pth/ml/{}.model".format(net_name)
            if os.path.exists(pth):
                return joblib.load("./pth/ml/{}.model".format(net_name)), True
        return RandomForestClassifier(random_state=10), False
    if net_name == "DF2":
        pth = "./pth/ml/{}.model".format(net_name)
        if os.path.exists(pth):
            return joblib.load("./pth/ml/{}.model".format(net_name)), True
        return CascadeForestClassifier(random_state=1,
                                       ), False
    if net_name == "DF":
        pth = "./pth/ml/{}.model".format(net_name)
        if os.path.exists(pth):
            return joblib.load("./pth/ml/{}.model".format(net_name)), True
        return CascadeForestClassifier(random_state=0,
                                       use_predictor=True,
                                       n_trees=550,
                                       n_estimators=4,
                                       ), False
    if net_name == "Xgboost":
        if not new:
            pth = "./pth/ml/{}.model".format(net_name)
            if os.path.exists(pth):
                return joblib.load("./pth/ml/{}.model".format(net_name)), True
        return xgb.XGBClassifier(), False

    if net_name == "TDFFM":
        net = torch.load('./pth/TMLAM/network.pth', map_location='cpu')
        return net, 400

    path = "./pth/{}/{}".format(coding_type, net_name)
    if not os.path.exists(path):
        os.makedirs(path)
    list = os.listdir(path)
    max = 0
    if _ep != 0:
        max = _ep
    else:
        for s in list:
            epoch = re.search(r'(\d+)$', s).group()
            if max < int(epoch):
                max = int(epoch)
    if max != 0:
        print("epoch:{}".format(max))
        net = torch.load('./pth/{}/{}/network.pth{}'.format(coding_type, net_name, max), map_location='cpu')
        return net, max
    if net_name == "deep1":
        net = deep1(1062, 512, 256, 128, 64, 16, 2)
        return net, 0
    if net_name == "lstm1":
        net = lstm1(seq_len, channel)
        return net, 0
    if net_name == "tf_d_4":
        net = tf_d_4()
        return net, 0

    print("err")
