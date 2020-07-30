# 生成 x_trian ,x_test y_train y_test
import numpy as np
datafile = 'data\\train.txt'

def save_npz():

    res_x = []
    res_y = []
    rest_x = []
    rest_y = []
  
    with open(datafile, 'r') as f:
        data = f.readlines()
        for i, line in enumerate(data):
            temp = []
            if i<1200:
                tmp = list(map(float, line.split(','))) #将str转为数值的类型
                for j in range(len(tmp)):
                    temp.append(tmp[j])
                res_x.append(temp)
                res_y.append(0)
                # print(type(res))
                x = np.array(res_x)
                y = np.array(res_y)
                # print(xyz.shape)
                # print(type(xyz))
                if (i+1)%4 ==0:
                    np.save('data/npy/x_train_' +str(i//4)  ,x)
                    np.save('data/npy/y_train_'+str(i//4) ,y)
                    res_x = []
                    res_y = []
            
            else:
                tmp = list(map(float, line.split(','))) #将str转为数值的类型
                for j in range(len(tmp)):
                    temp.append(tmp[j])
                rest_x.append(temp)
                rest_y.append(0)
                x = np.array(rest_x)
                y = np.array(rest_y)
                if (i+1)%4 ==0:
                    np.save('data/npy/x_test_' +str((i-1200)//4)  ,x)
                    np.save('data/npy/y_test_'+str((i-1200)//4) ,y)
                    rest_x = []
                    rest_y = []
            
                    


def read_npz():
    for i in range(300):
        file1 = 'data/npy/x_train_'+ str(i) +'.npy'
        file2 = 'data/npy/y_train_'+ str(i) +'.npy' 
        d1 = np.load(file1)
        d2 = np.load(file2)
        print(d1.shape, d2.shape)
    for j in range(97):
        file3 = 'data/npy/x_test_'+ str(j) +'.npy'
        file4 = 'data/npy/y_test_'+ str(j) +'.npy' 
        d3 = np.load(file3)
        d4 = np.load(file4)
        print(d3.shape, d4.shape)

# save_npz()
read_npz()
