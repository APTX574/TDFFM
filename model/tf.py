import numpy as np
import torch
from torch import nn


class tf(nn.Module):
    def __init__(self):
        super(tf, self).__init__()
        layer = nn.TransformerEncoderLayer(d_model=531, nhead=9)
        self.tf = nn.TransformerEncoder(layer, 4)
        self.lay1 = nn.Linear(531 * 5, 531)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(531, 256)
        self.lay3 = nn.Linear(256, 64)
        self.lay4 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.tf(x).reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x = self.lay3(x)
        x = self.ReLu(x)
        x = self.lay4(x)

        x = self.softmax(x)
        return x


class tf2(nn.Module):
    def __init__(self):
        super(tf2, self).__init__()
        layer = nn.TransformerEncoderLayer(d_model=531, nhead=9)
        self.tf = nn.TransformerEncoder(layer, 4)
        self.lstm = nn.LSTM(input_size=531, hidden_size=128)
        self.lay1 = nn.Linear(128 * 5, 531)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(531, 256)
        self.lay3 = nn.Linear(256, 64)
        self.lay4 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.tf(x)
        x = self.lstm(x)[0].reshape(x.shape[0], 128 * 5)
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x = self.lay3(x)
        x = self.ReLu(x)
        x = self.lay4(x)

        x = self.softmax(x)
        return x


class ca(nn.Module):
    # (batch_size,seq_len,channel)
    def __init__(self, input_dim=531):
        super(ca, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=5, out_channels=5 * 2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=5 * 2, out_channels=5, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.attention3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        x1 = x.transpose(1, 2)
        y1 = self.attention3(x1)
        y1 = y1.transpose(1, 2)

        return x * y * y1 + x


class tf3(nn.Module):
    def __init__(self):
        super(tf3, self).__init__()
        layer1 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        layer2 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        layer3 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        layer4 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        self.tf1 = nn.Sequential(nn.TransformerEncoder(layer1, 1), ca())
        self.tf2 = nn.Sequential(nn.TransformerEncoder(layer2, 1), ca())
        self.tf3 = nn.Sequential(nn.TransformerEncoder(layer3, 1), ca())
        self.tf4 = nn.Sequential(nn.TransformerEncoder(layer4, 1), ca())

        self.lstm = nn.LSTM(input_size=531, hidden_size=128, batch_first=True, dropout=0.5)
        self.ca = ca()
        self.lay1 = nn.Linear(128 * 5, 531)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(531, 256)
        self.lay3 = nn.Linear(256, 64)
        self.lay4 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x):

        x3 = self.tf3(x) + x
        x4 = self.tf4(x3) + x3
        x = self.lstm(x4)[0]
        x = x.reshape(x.shape[0], 128 * 5)
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x1 = self.lay3(x)
        x = self.ReLu(x1)
        x = self.lay4(x)

        x = self.softmax(x)
        return x, x1


class tf5(nn.Module):
    def __init__(self):
        super(tf5, self).__init__()
        layer1 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        layer2 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        layer3 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        layer4 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        self.tf1 = nn.Sequential(nn.TransformerEncoder(layer1, 1), ca())
        self.tf2 = nn.Sequential(nn.TransformerEncoder(layer2, 1), ca())
        self.tf3 = nn.Sequential(nn.TransformerEncoder(layer3, 1), ca())
        self.tf4 = nn.Sequential(nn.TransformerEncoder(layer4, 1), ca())

        self.lstm = nn.LSTM(input_size=531, hidden_size=128, batch_first=True, dropout=0.5)
        self.ca = ca()
        self.lay1 = nn.Linear(128 * 5, 531)
        self.ReLu = nn.ReLU()
        # self.nom=nn.BatchNorm1d()
        self.lay2 = nn.Linear(531, 256)
        self.lay3 = nn.Linear(256, 64)
        self.lay4 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x):

        x3 = self.tf3(x) + x
        x4 = self.tf4(x3) + x3
        x = self.lstm(x4)[0]
        x = x.reshape(x.shape[0], 128 * 5)
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x1 = self.lay3(x)
        x = self.ReLu(x1)
        x = self.lay4(x)

        x = self.softmax(x)
        return x


class tf4(nn.Module):
    def __init__(self):
        super(tf4, self).__init__()
        self.laybl = nn.Linear(100, 32)
        layer1 = nn.TransformerEncoderLayer(d_model=531, nhead=9)
        layer2 = nn.TransformerEncoderLayer(d_model=531, nhead=9)
        layer3 = nn.TransformerEncoderLayer(d_model=531, nhead=9)
        layer4 = nn.TransformerEncoderLayer(d_model=531, nhead=9)
        self.tf1 = nn.Sequential(nn.TransformerEncoder(layer1, 1), ca())
        self.tf2 = nn.Sequential(nn.TransformerEncoder(layer2, 1), ca())
        self.tf3 = nn.Sequential(nn.TransformerEncoder(layer3, 1), ca())
        self.tf4 = nn.Sequential(nn.TransformerEncoder(layer4, 1), ca())
        self.A1_1 = nn.Sequential(nn.Conv1d(in_channels=531, out_channels=120, kernel_size=1))
        self.A1_2 = nn.Sequential(nn.Conv1d(in_channels=531, out_channels=120, kernel_size=3, padding=1))
        self.A2_1 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=1))
        self.A2_2 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=3, padding=1))
        self.ca1 = ca()
        self.ca2 = ca()

        self.lay1 = nn.Linear((531 + 128) * 5, 531 * 5)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(531 * 5, 531)
        self.lay3 = nn.Linear(531, 64)
        self.lay4 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x_ = x.permute(0, 2, 1)
        x1 = self.A1_1(x_)  # (1,120,5)
        x2 = self.A1_2(x_)  # (1,120,5)
        x2 = torch.cat((x1, x2), 1)  # (1,240,5)
        x2 = x2.permute(0, 2, 1)
        x2 = x2 + x2 * self.ca1(x2)
        x2 = x2.permute(0, 2, 1)
        x3 = self.A2_1(x2)  # (1,64,5)
        x4 = self.A2_2(x2)  # (1,64,5)
        x2 = torch.cat((x3, x4), 1)  # (1,128,5)
        x2 = x2.permute(0, 2, 1)
        x2 = x2 + x2 * self.ca2(x2)

        x = self.tf1(x) + x
        x = self.tf2(x) + x
        x = self.tf3(x) + x
        x = self.tf4(x) + x

        x2 = torch.cat((x, x2), 2)
        x2 = x2.reshape(x2.shape[0], x2.shape[1] * x2.shape[2])
        x = self.lay1(x2)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.ReLu(x)
        x1 = self.lay4(x)

        x = self.softmax(x1)
        return x


class tf8(nn.Module):
    def __init__(self):
        super(tf8, self).__init__()
        layer1 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.0)
        layer2 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.0)
        layer3 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.0)
        layer4 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.0)
        self.tf1 = nn.Sequential(nn.TransformerEncoder(layer1, 1), ca())
        self.tf2 = nn.Sequential(nn.TransformerEncoder(layer2, 1), ca())
        self.tf3 = nn.Sequential(nn.TransformerEncoder(layer3, 1), ca())
        self.tf4 = nn.Sequential(nn.TransformerEncoder(layer4, 1), ca())
        self.conv1 = nn.Sequential(nn.Conv1d(5, 5, kernel_size=3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(5, 5, kernel_size=3, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(5, 5, kernel_size=3, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(5, 5, kernel_size=3, padding=1), nn.ReLU())

        self.lstm = nn.LSTM(input_size=20, bidirectional=True, hidden_size=32, dropout=0.0)

        self.lay1 = nn.Linear(595 * 5, 530)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(530, 256)
        self.lay3 = nn.Linear(256, 64)
        self.lay4 = nn.Linear(64, 16)
        self.lay5 = nn.Linear(16, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x, x_):


        x1 = self.tf1(x)
        x2 = self.tf2(x1) + x1
        x3 = self.tf3(x2) + x2
        x4 = self.tf4(x3) + x3
        x_2 = x_
        x_3 = self.conv1(x_2) + x_2
        x_4 = self.conv2(x_3) + x_3
        x_5 = self.conv3(x_4) + x_4
        x_6 = self.conv4(x_5) + x_5 + x_2
        x_6 = self.lstm(x_6)[0]
        x = torch.concat((x4, x_6), 2)
        x = x.reshape(x.shape[0], (531 + 64) * 5)

        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x1 = self.lay3(x)
        x = self.ReLu(x1)
        x = self.lay4(x)
        x = self.ReLu(x)
        x = self.lay5(x)

        x = self.softmax(x)
        return x


class tf_d(nn.Module):
    def __init__(self):
        super(tf_d, self).__init__()
        layer2_1 = nn.TransformerEncoderLayer(d_model=20, nhead=5, dropout=0.5)
        layer2_2 = nn.TransformerEncoderLayer(d_model=20, nhead=5, dropout=0.5)
        layer1_1 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        layer1_2 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        self.tf1_1 = nn.Sequential(nn.TransformerEncoder(layer1_1, 1), ca())
        self.tf1_2 = nn.Sequential(nn.TransformerEncoder(layer1_2, 1), ca())
        self.tf2_1 = nn.Sequential(nn.TransformerEncoder(layer2_1, 1), ca(20))
        self.tf2_2 = nn.Sequential(nn.TransformerEncoder(layer2_2, 1), ca(20))
        self.lstm = nn.LSTM(input_size=551, batch_first=True, hidden_size=128, dropout=0.5)
        self.lay1 = nn.Linear(128 * 5, 531)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(531, 256)
        self.lay3 = nn.Linear(256, 64)
        self.lay4 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x1, x2):
        x1_1 = self.tf1_1(x1) + x1
        x1_2 = self.tf1_2(x1_1) + x1_1 + x1
        x2_1 = self.tf2_1(x2) + x2
        x2_2 = self.tf2_2(x2_1) + x2_1 + x2
        x = torch.concat((x1_2, x2_2), 2)
        x = self.lstm(x)[0]
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x1 = self.lay3(x)
        x = self.ReLu(x1)
        x = self.lay4(x)
        x = self.softmax(x)
        return x
