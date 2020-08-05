import torch 
import argparse
import torch.nn as nn
import os
import numpy as np



# 660*4*4 -> 
class cnn(nn.Module):
    def __init__(self, input_size, output_size, channel, padding, kernel_size, stride, dropout):
        super(cnn,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size ,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(dropout)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512 ,256),
            nn.Sigmoid()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256 ,64),
            nn.Sigmoid()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(64 ,2),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=x.reshape(-1,256)
        x = self.fc1(x)
        # print("G网络第一层shape", x.shape)
        x = self.fc2(x)
        # print("G网络第二层shape", x.shape)
        # x = x.view(x.size(0),-1)
        # print(x.shape)
        x = x.reshape(20,32,10,10)
        x = self.layer1(x)
        # print("G网络第三层shape", x.shape)      
        x = self.layer2(x)
        # print("G网络第四层shape", x.shape)
        x = self.layer3(x)
        # print("G网络第五层shape", x.shape)
        return x