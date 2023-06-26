import random

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import loader
from tsne import plt_tsne
from utils.print_plt import print_plt
from utils.save import get_net


def change(x):
    seq_len = 5
    x = x.reshape((x.shape[0], seq_len, x.shape[1] // seq_len))
    x = x.transpose(2, 1)
    return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


setup_seed(20)


def main_test(seq_len=5, df_type="blosum62", use_gpu=False, net_name="TDFFM", batch_size=2000, channel=531,
              coding_type=None,
              new=True, a=0.433, is_p=True, tsne_=False):
    if coding_type is None:
        coding_type = ["aaindex", "blosum62"]
    score_list = []  # predict
    label_list = []  # label
    if tsne_:
        last_out = []

    # get datasets
    data = loader.DataSet(net_name, False, seq_len, channel, coding_type)
    dataloader = DataLoader(data, batch_size=batch_size)

    net, epoch = get_net(net_name, seq_len,
                         channel, coding_type[0], new)

    if use_gpu:
        net = net.cuda()
    net.eval()
    print(net)

    right_sum = 0
    if len(coding_type) == 2:
        net2 = joblib.load("pth/DeepForest/DF-{}.model".format(df_type))
        for i, (x1, x2, y) in enumerate(dataloader):
            if use_gpu:
                x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
            y2 = net2.predict_proba(x2)
            x1 = change(x1).transpose(1, 2)
            x2 = change(x2).transpose(1, 2)
            if tsne_ == 2:
                y1, y_m = net(x1, x2)
            else:
                y1 = net(x1, x2)
            y2 = torch.Tensor(y2)
            y_p = y1 * (a) + y2 * (1 - a)
            right_sum += (y_p.argmax(dim=1) == y).float().sum().item()
            score_tmp = y_p
            score_list.extend(score_tmp.detach().cpu().numpy())
            label_list.extend(y.cpu().numpy())
            if tsne_:
                last_out.append(y_m.detach().cpu().numpy())
    if tsne_:
        sub_out_list = last_out[0]
        for i in range(len(last_out) - 1):
            sub_out_list = np.concatenate((sub_out_list, last_out[i + 1]), 0)
        print(sub_out_list.shape)

        plt_tsne(sub_out_list, label_list, "tf3", "fin")
    print_plt(score_list, label_list, net_name,
              "{}-{}-a{}".format("TDFFM", epoch, a), is_p)


if __name__ == '__main__':
    main_test()
