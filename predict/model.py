import torch 
import argparse
import torch.nn as nn
import os
import numpy as np



# 660*4*4 -> 
class cnn(nn.Module):
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