import torch
from torch import nn



class deep1(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(deep1, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.sig = nn.PReLU()
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, n_hidden_4)
        self.layer5 = nn.Linear(n_hidden_4, n_hidden_5)

        self.layer6 = nn.Linear(n_hidden_5, out_dim)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.softmax(x)
        return x


class El(nn.Module):
    def __init__(self, conv_drpt=0.0, mlp_drpt=0.0, coord_channels=531, seq_len=5, one_hot_channels=20):
        super(El, self).__init__()

        #### MOTIF NET ####
        self.conv1 = nn.Conv1d(one_hot_channels, 16, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.penult1 = nn.Linear(320, 32)
        #### COORD NET ####
        self.mlp1 = nn.Linear(coord_channels * seq_len, coord_channels)
        self.bn1 = nn.BatchNorm1d(coord_channels)
        self.mlp2 = nn.Linear(coord_channels, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.mlp3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.penult2 = nn.Linear(128, 32)

        ### CAT LAYERS ###
        self.penult3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 16)
        self.out2 = nn.Linear(16, 2)
        self.softmax = nn.Softmax(1)

        #### MISC LAYERS ####
        self.relu = nn.ReLU()
        self.conv_drpt = nn.Dropout(p=conv_drpt)
        self.mlp_drpt = nn.Dropout(p=mlp_drpt)
        self.ablate = nn.Dropout(p=1.0)

    def forward(self, oneHot_motif, coords, version='seq-coord'):
        #### MOTIF NET ####
        conv1 = self.conv1(oneHot_motif.float())
        conv1 = self.relu(conv1)
        # conv1 = self.pool(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        # conv2 = self.pool(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        # conv3 = self.pool(conv3)

        seq_out = conv3.view(conv3.size()[0], -1)
        seq_out = self.penult1(seq_out)  ## SEQ PENULT
        seq_out = self.relu(seq_out)
        seq_out = self.conv_drpt(seq_out)

        #### COORD NET ####
        mlp1 = self.mlp1(coords)
        mlp1 = self.relu(mlp1)
        mlp1 = self.bn1(mlp1)
        mlp1 = self.mlp_drpt(mlp1)
        mlp2 = self.mlp2(mlp1)
        mlp2 = self.relu(mlp2)
        mlp2 = self.bn2(mlp2)
        mlp2 = self.mlp_drpt(mlp2)
        mlp3 = self.mlp3(mlp2)
        mlp3 = self.relu(mlp3)
        mlp3 = self.bn3(mlp3)
        mlp3 = self.mlp_drpt(mlp3)
        coord_out = self.penult2(mlp3)
        coord_out = self.relu(coord_out)

        if version == 'seq-coord':
            seq_out = self.conv_drpt(seq_out)  # seqCoord version
        else:
            seq_out = self.ablate(seq_out)  # seq-only version

        coords_out = self.mlp_drpt(coord_out)

        cat = torch.cat((seq_out, coords_out), 1)
        cat = self.penult3(cat)
        out = self.out(cat)
        out = self.out2(out)
        out = self.softmax(out)
        return out


