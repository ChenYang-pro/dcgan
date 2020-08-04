import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np

class convolution_d(nn.Module):
    def __init__(self, input_size, output_size, padding, kernel_size, stride, dropout):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size, 8, kernel_size,stride,padding),
            # nn.BatchNorm2d(8),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.SELU(8),
            nn.Dropout(dropout)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size,stride,padding),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.SELU(16),
            nn.Dropout(dropout)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size,stride,padding),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.SELU(32),
            nn.Dropout(dropout)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(3200 ,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(dropout)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256 ,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.layer1(x)
        # print("第一层shape:", x.shape)
        x = self.layer2(x)
        # print("第二层.shape:", x.shape)
        x = self.layer3(x)
        # print("第三层shape:", x.shape)
        # 卷积的输出拉伸为一行
        x = x.view(x.size(0),-1)
        # print("拉伸后shape:", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class convolution_g(nn.Module):
    def __init__(self, input_size, output_size, channel, padding, kernel_size, stride, dropout):
        super(convolution_g,self).__init__()
        self.fc1 = nn.Sequential(
            # nn.Linear(input_size,1280),
            nn.Linear(input_size,1*1280),
            nn.BatchNorm1d(1280),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.2)
            nn.Dropout(dropout)
        )

        self.fc2 = nn.Sequential(
            # nn.Linear(1280,3840),
            nn.Linear(1280,3200),
            nn.BatchNorm1d(3200),
            # nn.BatchNorm1d(3840),
            # nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size,stride,padding),
            nn.BatchNorm2d(16),
            # nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size,stride,padding),
            nn.BatchNorm2d(8),
            # nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size,stride,padding),
            nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Tanh()
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

class CausalConvDiscriminator(nn.Module):
    def __init__(self, input_size, output_size,  padding, kernel_size, stride, dropout):
        super().__init__()
        #Assuming same number of channels layerwise
        
        self.dcgan = convolution_d(input_size, output_size ,padding, kernel_size,stride, dropout)
        
    def forward(self, x):
        return self.dcgan(x)

class CausalConvGenerator(nn.Module):
    def __init__(self, input_size, output_size, channel, padding, kernel_size, stride, dropout):
        super().__init__()
        self.dcgan = convolution_g(input_size, output_size, channel, padding, kernel_size, stride, dropout)
        
    def forward(self, x):
        #return torch.sigmoid(self.tcn(x, channel_last))
        return self.dcgan(x)                                      # tanh -> sigmoid [0,1]


