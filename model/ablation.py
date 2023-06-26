import numpy as np
import torch
from torch import nn


class tf_d_0_0(nn.Module):
    def __init__(self):
        super(tf_d_0_0, self).__init__()
        layer2_1 = nn.TransformerEncoderLayer(d_model=20, nhead=5, dropout=0.5)
        layer2_2 = nn.TransformerEncoderLayer(d_model=20, nhead=5, dropout=0.5)
        layer1_1 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        layer1_2 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        self.tf1_1 = nn.Sequential(nn.TransformerEncoder(layer1_1, 1))
        self.tf1_2 = nn.Sequential(nn.TransformerEncoder(layer1_2, 1))
        self.tf2_1 = nn.Sequential(nn.TransformerEncoder(layer2_1, 1))
        self.tf2_2 = nn.Sequential(nn.TransformerEncoder(layer2_2, 1))
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
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x1 = self.lay3(x)
        x = self.ReLu(x1)
        x = self.lay4(x)
        x = self.softmax(x)
        return x


class tf_d_0(nn.Module):
    def __init__(self):
        super(tf_d_0, self).__init__()
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
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x1 = self.lay3(x)
        x = self.ReLu(x1)
        x = self.lay4(x)
        x = self.softmax(x)
        return x


class tf_d_1(nn.Module):
    def __init__(self):
        super(tf_d_1, self).__init__()
        self.lstm = nn.LSTM(input_size=551, batch_first=True, hidden_size=128, dropout=0.5)
        self.lay1 = nn.Linear(128 * 5, 531)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(531, 256)
        self.lay3 = nn.Linear(256, 64)
        self.lay4 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x1, x2):
        x = torch.concat((x1, x2), 2)
        x = self.lstm(x)[0]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x1 = self.lay3(x)
        x = self.ReLu(x1)
        x = self.lay4(x)
        x = self.softmax(x)
        return x


class tf_d_2(nn.Module):
    def __init__(self):
        super(tf_d_2, self).__init__()
        layer2_1 = nn.TransformerEncoderLayer(d_model=20, nhead=5, dropout=0.5)
        layer2_2 = nn.TransformerEncoderLayer(d_model=20, nhead=5, dropout=0.5)
        layer1_1 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        layer1_2 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        self.tf1_1 = nn.Sequential(nn.TransformerEncoder(layer1_1, 1), ca())
        self.tf1_2 = nn.Sequential(nn.TransformerEncoder(layer1_2, 1), ca())
        self.tf2_1 = nn.Sequential(nn.TransformerEncoder(layer2_1, 1), ca(20))
        self.tf2_2 = nn.Sequential(nn.TransformerEncoder(layer2_2, 1), ca(20))
        self.lay1 = nn.Linear(551 * 5, 551 * 2)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(551 * 2, 551)
        self.lay3 = nn.Linear(551, 256)
        self.lay4 = nn.Linear(256, 64)
        self.lay5 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x1, x2):
        x1_1 = self.tf1_1(x1) + x1
        x1_2 = self.tf1_2(x1_1) + x1_1 + x1
        x2_1 = self.tf2_1(x2) + x2
        x2_2 = self.tf2_2(x2_1) + x2_1 + x2
        x = torch.concat((x1_2, x2_2), 2)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x = self.lay3(x)
        x = self.ReLu(x)
        x = self.lay4(x)
        x = self.ReLu(x)
        x = self.lay5(x)
        x = self.softmax(x)
        return x
        # return x


class tf_d_3(nn.Module):
    def __init__(self):
        super(tf_d_3, self).__init__()

        self.lay1 = nn.Linear(551 * 5, 551 * 2)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(551 * 2, 551)
        self.lay3 = nn.Linear(551, 256)
        self.lay4 = nn.Linear(256, 64)
        self.lay5 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x1, x2):
        x = torch.concat((x1, x2), 2)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x = self.lay3(x)
        x = self.ReLu(x)
        x = self.lay4(x)
        x = self.ReLu(x)
        x = self.lay5(x)
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


class tf_d_4(nn.Module):
    def __init__(self):
        super(tf_d_4, self).__init__()
        layer2_1 = nn.TransformerEncoderLayer(d_model=20, nhead=5, dropout=0.5)
        layer1_1 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        self.tf1_1 = nn.Sequential(nn.TransformerEncoder(layer1_1, 1), ca())
        self.tf2_1 = nn.Sequential(nn.TransformerEncoder(layer2_1, 1), ca(20))
        self.lstm = nn.LSTM(input_size=551, batch_first=True, hidden_size=128, dropout=0.5)
        self.lay1 = nn.Linear(128 * 5, 531)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(531, 256)
        self.lay3 = nn.Linear(256, 64)
        self.lay4 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x1, x2):
        x1_1 = self.tf1_1(x1) + x1
        x2_1 = self.tf2_1(x2) + x2
        x = torch.concat((x1_1, x2_1), 2)
        x = self.lstm(x)[0]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x1 = self.lay3(x)
        x = self.ReLu(x1)
        x = self.lay4(x)
        x = self.softmax(x)
        return x


class tf_d_w1_a(nn.Module):
    def __init__(self):
        super(tf_d_w1_a, self).__init__()
        layer1_1 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        layer1_2 = nn.TransformerEncoderLayer(d_model=531, nhead=9, dropout=0.5)
        self.tf1_1 = nn.Sequential(nn.TransformerEncoder(layer1_1, 1), ca())
        self.tf1_2 = nn.Sequential(nn.TransformerEncoder(layer1_2, 1), ca())
        self.lstm = nn.LSTM(input_size=531, batch_first=True, hidden_size=128, dropout=0.5)
        self.lay1 = nn.Linear(128 * 5, 531)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(531, 256)
        self.lay3 = nn.Linear(256, 64)
        self.lay4 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x1):
        x1_1 = self.tf1_1(x1) + x1
        x1_2 = self.tf1_2(x1_1) + x1_1
        x = self.lstm(x1_2)[0]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x1 = self.lay3(x)
        x = self.ReLu(x1)
        x = self.lay4(x)
        x = self.softmax(x)
        return x


class tf_d_w1_b(nn.Module):
    def __init__(self):
        super(tf_d_w1_b, self).__init__()
        layer1_1 = nn.TransformerEncoderLayer(d_model=20, nhead=5, dropout=0.5)
        layer1_2 = nn.TransformerEncoderLayer(d_model=20, nhead=5, dropout=0.5)
        self.tf1_1 = nn.Sequential(nn.TransformerEncoder(layer1_1, 1), ca(20))
        self.tf1_2 = nn.Sequential(nn.TransformerEncoder(layer1_2, 1), ca(20))
        self.lstm = nn.LSTM(input_size=20, batch_first=True, hidden_size=20, dropout=0.5)
        self.lay1 = nn.Linear(20 * 5, 64)
        self.ReLu = nn.ReLU()
        self.lay2 = nn.Linear(64, 32)
        self.lay3 = nn.Linear(32, 16)
        self.lay4 = nn.Linear(16, 2)

        self.softmax = nn.Softmax(1)

    def forward(self, x1):
        x1_1 = self.tf1_1(x1) + x1
        x1_2 = self.tf1_2(x1_1) + x1_1 + x1
        x = self.lstm(x1_2)[0]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.lay1(x)
        x = self.ReLu(x)
        x = self.lay2(x)
        x = self.ReLu(x)
        x1 = self.lay3(x)
        x = self.ReLu(x1)
        x = self.lay4(x)
        x = self.softmax(x)
        return x
