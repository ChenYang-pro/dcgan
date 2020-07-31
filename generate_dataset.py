# 生成 x_trian ,x_test y_train y_test
import numpy as np
datafile = 'data/origin_data/train201801.txt'

def strip_space(fname,wname):
    with open(fname,'r') as f:
        lines = f.readline()
        while lines:
            with open(wname,'a') as f1:
                if lines =="\n":
                    lines = f.readline()
                else:
                    f1.write(lines)
                    lines = f.readline()


def save_npz():

    res_x = []
    res_y = []
    rest_x = []
    rest_y = []
  
    with open(datafile, 'r',encoding='UTF-8') as f:
        data = f.readlines()
        for i, line in enumerate(data):
            temp = []
            if i<2640:
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
                    np.save('data/npz/x_train_' +str(i//4)  ,x)
                    np.save('data/npz/y_train_'+str(i//4) ,y)
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
                    np.save('data/npz/x_test_' +str((i-2640)//4)  ,x)
                    np.save('data/npz/y_test_'+str((i-2640)//4) ,y)
                    rest_x = []
                    rest_y = []
            
                    


def read_npz():
    file1 = 'data/npz/y_train_0.npy'
    file2 = 'data/npz/y_train_659.npy' 
    file3 = 'data/npz/y_test_0.npy' 
    file4 = 'data/npz/y_test_134.npy' 
    d1 = np.load(file1)
    d2 = np.load(file2)
    d3 = np.load(file3)
    d4 = np.load(file4)
    print(d1)
    print(d2)
    print(d3)
    print(d4)
    # for i in range(660):
    #     # file1 = 'data/npz/x_train_'+ str(i) +'.npy'
    #     # file2 = 'data/npz/y_train_'+ str(i) +'.npy' 
    #     d1 = np.load(file1)
    #     d2 = np.load(file2)
    #     print(d1.shape, d2.shape)
    # for j in range(135):
    #     file3 = 'data/npz/x_test_'+ str(j) +'.npy'
    #     file4 = 'data/npz/y_test_'+ str(j) +'.npy' 
    #     d3 = np.load(file3)
    #     d4 = np.load(file4)
    #     print(d3.shape, d4.shape)

if __name__ == "__main__":
    # 去空格 
    # filename1 = 'data\origin_data/5-I80E_2018.1.14-31_incident_matrix.txt'
    # filename2 = 'data\origin_data/5-I80W_2018.1.14-31_incident_matrix.txt'
    # writename = 'data/origin_data/train.txt'
    # strip_space(filename1,writename)
    # strip_space(filename2,writename)
    # save_npz()
    read_npz()
