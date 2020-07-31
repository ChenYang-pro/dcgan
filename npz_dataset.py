import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
# 做数据集

class npz_Dataset(Dataset):
    """Btp time series dataset."""
    def __init__(self, root_dir,dim_len,SEQ_len):
        """
        Args:
            root_dir (string): path to npy file
            SEQ_len: reshape data to time_step form
        """
        self.root_dir = root_dir
        # self.CITY = CITY
        self.SEQ_len = SEQ_len                                 # 8
        #self.category = category
        self.dim_len = dim_len
        
        x_train = self.load_data()
        x_train = self.reshape(x_train).astype(np.float32)     # object->float
        print(x_train.shape)

        self.data = torch.from_numpy(x_train).float()          # float type  (samples, steps 8,features)
        self.seq_len = self.data.size(1)                       # 必要么？？？？？？？？？？？？
        # print("data.size(1):",self.seq_len)




    def load_data(self,category=None, with_geocode=False):   ## ?? with_geocode
        x_train_temp =  np.load(self.root_dir + '/x_train_0.npy', allow_pickle=True)
        y_train_temp = np.load(self.root_dir + '/y_train_0.npy', allow_pickle=True)
        x_test_temp = np.load(self.root_dir + '/x_test_0.npy', allow_pickle=True)
        y_test_temp = np.load(self.root_dir + '/y_test_0.npy', allow_pickle=True)
        X_train = np.array(x_train_temp)
        Y_train = np.array(y_train_temp)
        X_test = np.array(x_test_temp)
        Y_test = np.array(y_test_temp)
        for i in range(1,660):
            x_train = np.load(self.root_dir + '/x_train_'+ str(i) +'.npy', allow_pickle=True)
            X_train = np.vstack((X_train,x_train))
            y_train = np.load(self.root_dir + '/y_train_'+ str(i) +'.npy', allow_pickle=True)
            Y_train = np.vstack((Y_train,y_train))
            # X_data = np.concatenate([X_train], axis=0)     
            # y_data = np.concatenate([y_train]) 
            # X_data = np.concatenate([X_train,X_test], axis=0)
        for j in range(1,140):
            x_test = np.load(self.root_dir +'/x_test_'+ str(j) +'.npy' ,allow_pickle=True)
            X_test = np.vstack((X_test,x_test))
            y_test = np.load(self.root_dir +'/y_test_'+ str(j) +'.npy' ,allow_pickle=True)
            Y_test = np.vstack((Y_test,y_test))
            # X_data = np.concatenate([X_test], axis=0)  
            # y_data = np.concatenate([y_test])
        # if not with_geocode:
        #     X_train = X_train[:, 0:-1]
        #     X_test = X_test[:, 0:-1]

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


        # indices = y_data==1
        # X_data = X_data[indices]

        # print(X_data.shape)

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
        # reshape 800*1*4*4
        x = x.reshape((int(x.shape[0] / self.dim_len), int(x.shape[1] / self.SEQ_len), self.dim_len, self.SEQ_len))
        print(x.shape)
        return x


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        if not hasattr(self, 'max') or not hasattr(self, 'min'):
            raise Exception("You are calling denormalize, but the input was not normalized")
        return 0.5 * (x*self.max - x*self.min + self.max + self.min)

# import numpy as np

# file1 = 'DAP\data\\train_set\X_test_Houston.npy'
# df = np.load(file1, allow_pickle= True)
# print(df.shape)

# file2 = 'DAP\data\\train_set\y_test_Houston.npy'
# de = np.load(file2, allow_pickle= True)
# print(de.shape)
# print(de)