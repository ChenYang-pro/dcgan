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
            temp1 = []
            # temp = []
            if i<2640:
                tmp = list(map(float, line.split(','))) #将str转为数值的类型
                for j in range(len(tmp)):
                    temp.append(tmp[j])
                res_x.append(temp)
                res_y.append(0)
                # print(type(res))
                
                # print(xyz.shape)
                # print(type(xyz))
                # if (i+1)%4 ==0:
                #     np.save('data/npz/x_train_' +str(i//4)  ,x)
                #     np.save('data/npz/y_train_'+str(i//4) ,y)
                #     res_x = []
                #     res_y = []
                # np.save('data/train_set/x_train' ,x)
                # np.save('data/train_set/y_train',y)
            else:
                tmp = list(map(float, line.split(','))) #将str转为数值的类型
                for j in range(len(tmp)):
                    temp1.append(tmp[j])
                rest_x.append(temp1)
                rest_y.append(0)
                # x = np.array(rest_x)
                # y = np.array(rest_y)
                # if (i+1)%4 ==0:
                #     np.save('data/npz/x_test_' +str((i-2640)//4)  ,x)
                #     np.save('data/npz/y_test_'+str((i-2640)//4) ,y)
                # np.save('data/train_set/x_test' ,x)
                # np.save('data/train_set/y_test' ,y)
            
        xtr = np.array(res_x)
        ytr = np.array(res_y)
        np.save('data/train_set/x_train' ,xtr)
        np.save('data/train_set/y_train',ytr)
        xte = np.array(rest_x)
        yte = np.array(rest_y)
        np.save('data/train_set/x_test' ,xte)
        np.save('data/train_set/y_test' ,yte)

def read_npz():
    file1 = 'data/train_set/x_train.npy'
    file2 = 'data/train_set/y_train.npy' 
    file3 = 'data/train_set/x_test.npy' 
    file4 = 'data/train_set/y_test.npy' 
    d1 = np.load(file1)
    d2 = np.load(file2)
    d3 = np.load(file3)
    d4 = np.load(file4)
    print(d1.shape)
    print(d2.shape)
    print(d3.shape)
    print(d4.shape)
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
    # 去空格 ``
    # filename1 = 'data\origin_data/5-I80E_2018.1.14-31_incident_matrix.txt'
    # filename2 = 'data\origin_data/5-I80W_2018.1.14-31_incident_matrix.txt'
    # writename = 'data/origin_data/train.txt'
    # strip_space(filename1,writename)
    # strip_space(filename2,writename)
    save_npz()
    read_npz()
