import torch.utils.data as Data

from dataset import load_dataset


class DataSet(Data.Dataset):
    def __init__(self, net_name, is_train, sql_len, channels, coding_types):
        super(DataSet, self).__init__()
        self.is_train = is_train
        self.net_name = net_name
        self.coding_type = coding_types
        if is_train:
            self.train = load_dataset.load(coding_types)[0]
        else:
            self.train = load_dataset.load(coding_types)[1]
        print(self.__len__())

    def __getitem__(self, index):
        len_x = len(self.train.data)
        if len_x == 2:
            return self.train.data[self.coding_type[0]][index], \
                   self.train.data[self.coding_type[1]][index], \
                   self.train.target[index]
        if len_x == 1:
            return self.train.data[self.coding_type[0]][index], \
                   self.train.target[index]
        if len_x == 3:
            return self.train.data[self.coding_type[0]][index], \
                   self.train.data[self.coding_type[1]][index], \
                   self.train.data[self.coding_type[2]][index], \
                   self.train.target[index]

    def __len__(self):
        return self.train.target.shape[0]
