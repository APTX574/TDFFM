import torch
from torch import nn

from dataset.load_dataset import load
import utils.save as save


class hy(nn.Module):
    def __init__(self):
        super(hy, self).__init__()
        self.lstm = nn.Linear(in_features=4, out_features=4)
        self.ReLu = nn.ReLU()
        self.lstm1 = nn.Linear(in_features=4, out_features=2)
        self.softmax = nn.Softmax(1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.lstm(x)
        x = self.lstm1(x)
        x = self.softmax(x)
        return x

    def get_bz(self):
        att = "None"
        coding_type = ["aaindex", "one_hot", "blosum62"]
        return coding_type, att


class lstm1(nn.Module):

    def __init__(self, seq_len, channel):
        super(lstm1, self).__init__()
        self.A1_1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=1))
        self.A1_2 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=3, padding=1))
        self.A2_1 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=1))
        self.A2_2 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=3, padding=1))
        self.pooling1 = nn.MaxPool1d(kernel_size=3)

        self.lstm = nn.LSTM(input_size=128, hidden_size=64)
        self.lay1 = nn.Linear(in_features=64 * seq_len, out_features=32)
        self.lay2 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x1 = self.A1_1(x)  # (1,120,5)
        x2 = self.A1_2(x)  # (1,120,5)
        x = torch.cat((x1, x2), 1)  # (1,240,5)
        x1 = self.A2_1(x)  # (1,64,5)
        x2 = self.A2_2(x)  # (1,64,5)
        x = torch.cat((x1, x2), 1)  # (1,128,5)
        x = self.lstm(x.transpose(1, 2))[0]  # (1,32,5)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # (1,320)
        x = self.lay1(x)  # (1,32)
        x = self.lay2(x)  # (1,2)
        x = self.softmax(x)
        return x

    def get_bz(self):
        att = "None"
        coding_type = ["aaindex"]
        return coding_type, att


class lstm2(nn.Module):

    def __init__(self, seq_len, channel):
        super(lstm2, self).__init__()
        self.A1_1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=1), nn.BatchNorm1d(120))
        self.A1_2 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=3, padding=1),
                                  nn.BatchNorm1d(120))
        self.A2_1 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=1), nn.BatchNorm1d(64))
        self.A2_2 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=3, padding=1),
                                  nn.BatchNorm1d(64))
        self.pooling1 = nn.MaxPool1d(kernel_size=3)
        self.A3_1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1), nn.BatchNorm1d(32))
        self.A3_2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
                                  nn.BatchNorm1d(32))
        self.pooling1 = nn.MaxPool1d(kernel_size=3)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64)
        self.lay1 = nn.Linear(in_features=64 * seq_len, out_features=32)
        self.lay2 = nn.Linear(in_features=32, out_features=8)
        self.lay3 = nn.Linear(in_features=8, out_features=2)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x1 = self.A1_1(x)  # (1,120,5)
        x2 = self.A1_2(x)  # (1,120,5)
        x = torch.cat((x1, x2), 1)  # (1,240,5)
        x1 = self.A2_1(x)  # (1,64,5)
        x2 = self.A2_2(x)  # (1,64,5)

        x = torch.cat((x1, x2), 1)  # (1,128,5)
        x1 = self.A3_1(x)  # (1,64,5)
        x2 = self.A3_2(x)  # (1,64,5)

        x = torch.cat((x1, x2), 1)  # (1,128,5)
        x = self.lstm(x.transpose(1, 2))[0]  # (1,32,5)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # (1,320)
        x = self.lay1(x)  # (1,32)
        x = self.lay2(x)  # (1,2)
        x = self.lay3(x)  # (1,2)
        x = self.softmax(x)
        return x

    def get_bz(self):
        att = "None"
        coding_type = ["aaindex"]
        return coding_type, att


class lstm3(nn.Module):

    def __init__(self, seq_len=5, channel=531):
        super(lstm3, self).__init__()
        self.para = torch.nn.Parameter(torch.rand(channel).requires_grad_())
        self.A1_1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=1))
        self.A1_2 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=3, padding=1),
                                  )
        self.A2_1 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=1))
        self.A2_2 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=3, padding=1),
                                  )
        self.pooling1 = nn.MaxPool1d(kernel_size=3)

        self.lstm = nn.LSTM(input_size=128, hidden_size=64)
        self.lay1 = nn.Linear(in_features=64 * seq_len, out_features=32, )
        self.lay2 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.mul(x, self.para)
        x = x.permute(0, 2, 1)
        x1 = self.A1_1(x)  # (1,120,5)
        x2 = self.A1_2(x)  # (1,120,5)
        x = torch.cat((x1, x2), 1)  # (1,240,5)
        x1 = self.A2_1(x)  # (1,64,5)
        x2 = self.A2_2(x)  # (1,64,5)
        x = torch.cat((x1, x2), 1)  # (1,128,5)
        x = self.lstm(x.transpose(1, 2))[0]  # (1,32,5)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # (1,320)
        x = self.lay1(x)  # (1,32)
        x = self.lay2(x)  # (1,2)
        x = self.softmax(x)
        return x

    def get_bz(self):
        att = "aaindex_att"
        coding_type = ["aaindex"]
        return coding_type, att


class lstm4(nn.Module):

    def __init__(self, seq_len, channel, ):
        super(lstm4, self).__init__()

        self.para2 = torch.nn.Parameter(torch.rand(5).requires_grad_())
        self.conv1 = nn.Conv1d(20, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.penult1 = nn.Linear(320, 32)

        self.para = torch.nn.Parameter(torch.rand(channel).requires_grad_())
        self.A1_1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=1))
        self.A1_2 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=3, padding=1),
                                  )
        self.A2_1 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=1))
        self.A2_2 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=3, padding=1),
                                  )
        self.pooling1 = nn.MaxPool1d(kernel_size=3)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=128, hidden_size=64)
        self.lay1 = nn.Linear(in_features=64 * seq_len, out_features=32, )
        self.lay2 = nn.Linear(in_features=64, out_features=32)
        self.lay3 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(1)

    def forward(self, x, x3):
        x3 = x3 + torch.mul(x3, self.para2)
        conv1 = self.conv1(x3.float())
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        seq_out = conv3.view(conv3.size()[0], -1)
        seq_out = self.penult1(seq_out)  ## SEQ PENULT

        # x = self.A0_1(x)  # (1,channel,5)
        x = x.permute(0, 2, 1)
        x = x + torch.mul(x, self.para)
        x = x.permute(0, 2, 1)
        x1 = self.A1_1(x)  # (1,120,5)
        x2 = self.A1_2(x)  # (1,120,5)
        x = torch.cat((x1, x2), 1)  # (1,240,5)
        x1 = self.A2_1(x)  # (1,64,5)
        x2 = self.A2_2(x)  # (1,64,5)
        x = torch.cat((x1, x2), 1)  # (1,128,5)
        x = self.lstm(x.transpose(1, 2))[0]  # (1,32,5)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # (1,320)
        x = self.lay1(x)  # (1,32)
        x = torch.cat((seq_out, x), 1)
        x = self.lay2(x)  # (1,2)
        x = self.lay3(x)  # (1,2)
        x = self.softmax(x)
        return x

    def get_bz(self):
        att = "aaindex_att,one_att"
        coding_type = ["aaindex", "one_hot"]
        return coding_type, att


class lstm5(nn.Module):

    def __init__(self, seq_len, channel, ):
        super(lstm5, self).__init__()

        self.conv1 = nn.Conv1d(20, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.penult1 = nn.Linear(320, 32)

        self.para = torch.nn.Parameter(torch.rand(channel).requires_grad_())
        self.A1_1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=1))
        self.A1_2 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=3, padding=1),
                                  )
        self.A2_1 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=1))
        self.A2_2 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=3, padding=1),
                                  )
        self.pooling1 = nn.MaxPool1d(kernel_size=3)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=128, hidden_size=64)
        self.lay1 = nn.Linear(in_features=64 * seq_len, out_features=32, )
        self.lay2 = nn.Linear(in_features=64, out_features=32)
        self.lay3 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(1)

    def forward(self, x, x3):
        conv1 = self.conv1(x3.float())
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        seq_out = conv3.view(conv3.size()[0], -1)
        seq_out = self.penult1(seq_out)  ## SEQ PENULT

        x = x.permute(0, 2, 1)
        x = x + torch.mul(x, self.para)
        x = x.permute(0, 2, 1)
        x1 = self.A1_1(x)  # (1,120,5)
        x2 = self.A1_2(x)  # (1,120,5)
        x = torch.cat((x1, x2), 1)  # (1,240,5)
        x1 = self.A2_1(x)  # (1,64,5)
        x2 = self.A2_2(x)  # (1,64,5)
        x = torch.cat((x1, x2), 1)  # (1,128,5)
        x = self.lstm(x.transpose(1, 2))[0]  # (1,32,5)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # (1,320)
        x = self.lay1(x)  # (1,32)
        x = torch.cat((seq_out, x), 1)
        x = self.lay2(x)  # (1,2)
        x = self.lay3(x)  # (1,2)
        x = self.softmax(x)
        return x

    def get_bz(self):
        att = "aaindex_att"
        coding_type = ["aaindex", "one_hot"]
        return coding_type, att


class lstm6(nn.Module):

    def __init__(self, seq_len, channel, ):
        super(lstm6, self).__init__()

        self.para2 = torch.nn.Parameter(torch.rand(5).requires_grad_())
        self.conv1 = nn.Conv1d(20, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.penult1 = nn.Linear(320, 32)
        self.relu = nn.ReLU()
        self.A1_1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=1))
        self.A1_2 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=3, padding=1),
                                  )
        self.A2_1 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=1))
        self.A2_2 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=3, padding=1),
                                  )
        self.pooling1 = nn.MaxPool1d(kernel_size=3)

        self.lstm = nn.LSTM(input_size=128, hidden_size=64)
        self.lay1 = nn.Linear(in_features=64 * seq_len, out_features=32, )
        self.lay2 = nn.Linear(in_features=64, out_features=32)
        self.lay3 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(1)

    def forward(self, x, x3):
        x3 = x3 + torch.mul(x3, self.para2)
        conv1 = self.conv1(x3.float())
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        seq_out = conv3.view(conv3.size()[0], -1)
        seq_out = self.penult1(seq_out)  ## SEQ PENULT

        x1 = self.A1_1(x)  # (1,120,5)
        x2 = self.A1_2(x)  # (1,120,5)
        x = torch.cat((x1, x2), 1)  # (1,240,5)
        x1 = self.A2_1(x)  # (1,64,5)
        x2 = self.A2_2(x)  # (1,64,5)
        x = torch.cat((x1, x2), 1)  # (1,128,5)
        x = self.lstm(x.transpose(1, 2))[0]  # (1,32,5)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # (1,320)
        x = self.lay1(x)  # (1,32)
        x = torch.cat((seq_out, x), 1)
        x = self.lay2(x)  # (1,2)
        x = self.lay3(x)  # (1,2)
        x = self.softmax(x)
        return x

    def get_bz(self):
        att = "one_att"
        coding_type = ["aaindex", "one_hot"]
        return coding_type, att


class lstm7(nn.Module):

    def __init__(self, seq_len, channel, ):
        super(lstm7, self).__init__()

        self.conv1 = nn.Conv1d(20, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.penult1 = nn.Linear(320, 32)
        self.relu = nn.ReLU()
        self.A1_1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=1))
        self.A1_2 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=3, padding=1),
                                  )
        self.A2_1 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=1))
        self.A2_2 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=3, padding=1),
                                  )
        self.pooling1 = nn.MaxPool1d(kernel_size=3)

        self.lstm = nn.LSTM(input_size=128, hidden_size=64)
        self.lay1 = nn.Linear(in_features=64 * seq_len, out_features=32, )
        self.lay2 = nn.Linear(in_features=64, out_features=32)
        self.lay3 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(1)

    def forward(self, x, x3):
        conv1 = self.conv1(x3.float())
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        seq_out = conv3.view(conv3.size()[0], -1)
        seq_out = self.penult1(seq_out)  ## SEQ PENULT

        # x = self.A0_1(x)  # (1,channel,5)
        x1 = self.A1_1(x)  # (1,120,5)
        x2 = self.A1_2(x)  # (1,120,5)
        x = torch.cat((x1, x2), 1)  # (1,240,5)
        x1 = self.A2_1(x)  # (1,64,5)
        x2 = self.A2_2(x)  # (1,64,5)
        x = torch.cat((x1, x2), 1)  # (1,128,5)
        x = self.lstm(x.transpose(1, 2))[0]  # (1,32,5)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # (1,320)
        x = self.lay1(x)  # (1,32)
        x = torch.cat((seq_out, x), 1)
        x = self.lay2(x)  # (1,2)
        x = self.lay3(x)  # (1,2)
        x = self.softmax(x)
        return x

    def get_bz(self):
        att = "None"
        coding_type = ["aaindex", "one_hot"]
        return coding_type, att


class lstm8(nn.Module):

    def __init__(self, seq_len, channel, ):
        super(lstm8, self).__init__()

        self.conv1 = nn.Conv1d(20, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.penult1 = nn.Linear(320, 32)
        self.relu = nn.ReLU()
        self.A1_1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=1))
        self.A1_2 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=120, kernel_size=3, padding=1),
                                  )
        self.A2_1 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=1))
        self.A2_2 = nn.Sequential(nn.Conv1d(in_channels=240, out_channels=64, kernel_size=3, padding=1),
                                  )
        self.pooling1 = nn.MaxPool1d(kernel_size=3)

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True)
        self.lay1 = nn.Linear(in_features=64 * seq_len * 2, out_features=120, )
        self.lay1_1 = nn.Linear(in_features=120, out_features=32, )
        self.lay2 = nn.Linear(in_features=64, out_features=32)
        self.lay3 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(1)

    def forward(self, x, x3):
        conv1 = self.conv1(x3.float())
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        seq_out = conv3.view(conv3.size()[0], -1)
        seq_out = self.penult1(seq_out)  ## SEQ PENULT

        x1 = self.A1_1(x)  # (1,120,5)
        x2 = self.A1_2(x)  # (1,120,5)
        x = torch.cat((x1, x2), 1)  # (1,240,5)
        x1 = self.A2_1(x)  # (1,64,5)
        x2 = self.A2_2(x)  # (1,64,5)
        x = torch.cat((x1, x2), 1)  # (1,128,5)
        x = self.lstm(x.transpose(1, 2))[0]  # (1,32,5)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # (1,320)
        # print(x.shape)
        x = self.lay1(x)  # (1,32)
        x = self.lay1_1(x)  # (1,32)
        x = torch.cat((seq_out, x), 1)
        x = self.lay2(x)  # (1,2)
        x = self.lay3(x)  # (1,2)
        x = self.softmax(x)
        return x

    def get_bz(self):
        att = "None"
        coding_type = ["aaindex", "one_hot"]
        return coding_type, att
