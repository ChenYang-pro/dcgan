import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
# 做数据集

class AccidentDataset(Dataset):
    """Btp time series dataset."""
    def __init__(self, root_dir,CITY,SEQ_len,category):
        """
        Args:
            root_dir (string): path to npy file
            CITY: path to npy file compare with root_dir
            SEQ_len: reshape data to time_step form
            category：subset of data, array form
        """
        self.root_dir = root_dir
        self.CITY = CITY
        self.SEQ_len = SEQ_len                                 # 8
        #self.category = category

        x_train = self.load_data(category)
        x_train = self.reshape(x_train).astype(np.float32)     # object->float
        print(x_train.shape)

        self.data = torch.from_numpy(x_train).float()          # float type  (samples, steps 8,features)
        self.seq_len = self.data.size(1)                       # 必要么？？？？？？？？？？？？





    def load_data(self,category=None, with_geocode=False):

        X_train = np.load(self.root_dir + 'train_set/X_train_' + self.CITY + '.npy', allow_pickle=True)
        X_test = np.load(self.root_dir + 'train_set/X_test_' + self.CITY + '.npy', allow_pickle=True)
        y_train = np.load(self.root_dir +'train_set/y_train_' + self.CITY + '.npy',allow_pickle=True)
        y_test = np.load(self.root_dir +'train_set/y_test_' + self.CITY + '.npy',allow_pickle=True)

        if not with_geocode:
            X_train = X_train[:, 0:-1]
            X_test = X_test[:, 0:-1]

        if category != None:
            l_train = []
            l_test = []
            for cat in category:
                l_train.append(self.reshape_cat(X_train, cat))
                l_test.append(self.reshape_cat(X_test, cat))
            X_train = np.concatenate(l_train, axis=1)
            X_test = np.concatenate(l_test, axis=1)


        X_data = np.concatenate([X_train,X_test], axis=0)                     # no need for split train and test
        y_data = np.concatenate([y_train,y_test])
        print(X_data.shape)


        indices = y_data==1
        X_data = X_data[indices]

        print(X_data.shape)

        return X_data



    def reshape_cat(self,array, category):
        l = []
        b = array[:, 0:-14]
        if category != 'geohash' and category != 'NLP':
            for i in range(self.SEQ_len):
                c = b[:, i * 25:i * 25 + 25]
                if category == 'traffic':
                    # d = np.concatenate((c[:,0:9],c[:,-5:]),axis=1)
                    d = np.concatenate([c[:, 1:2], c[:, 3:10]], axis=1)
                elif category == 'weather':
                    d = c[:, 10:-5]
                elif category == 'time':
                    d = np.concatenate([c[:, 0:1], c[:, 2:3], c[:, -5:]], axis=1)
                else:
                    d = c
                l.append(d)
            n = np.concatenate(l, axis=1)
            # if category!='no_geohash':
            #    return np.concatenate((n,array[:,-14:]),axis=1)
            return n
        elif category == 'NLP':
            return array[:, -100:]
        else:
            return array[:, -114:-100]

    def reshape(self,x):
        # x = x[:,0:-114]
        x = x.reshape((x.shape[0], self.SEQ_len, int(x.shape[1] / self.SEQ_len)))
        return x


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    dataset = AccidentDataset('/content/DAP/data/','LosAngeles',8,['traffic','weather','time'])
    # 断言 当判断条件为false时执行 if not dataset : raise AssertionError
    assert dataset
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,shuffle=True, num_workers=int(2))
    import datetime
    starttime = datetime.datetime.now()

    for i, data in enumerate(dataloader, 0):
      if i==0:
        print(i,data.shape)
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)

    