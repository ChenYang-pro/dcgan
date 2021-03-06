import torch 
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pth', default="checkpoints/08_01_06_51I80_netG_epoch_49.pth", help='path to .pth')
parser.add_argument('--savepth', default="generate_data", help='generate_data')
parser.add_argument('--looptimes', default=40, help='generate_data num=looptime*20')
opt = parser.parse_args()

def generate():
    for i in range(int(opt.looptimes)):
        x = torch.randn(20,1,256)
        # print(x)
        netG = torch.load(opt.pth)
        # tensor to numpy
        data =netG(x).reshape(20,4,4).detach().numpy()

        # da = data.numpy()
        print("==========================")
        print(data[0])
        # print(type(data))
        # print(data.shape)
        # save data
        ti = time.strftime("%m-%d-%H:%M")
        np.save(str(opt.savepth) + '/G_data_' + ti + str(i),data)

if __name__ == "__main__":
    generate()



