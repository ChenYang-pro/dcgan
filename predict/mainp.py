import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import datetime
from utils import time_series_to_plot
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time 
from prenpz_dataset import npz_Dataset
from model import cnn
# 命令行参数

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="npz", help='dataset to use (only btp for now)')
parser.add_argument('--pth', default="checkpoints/08_01_06_51I80_netG_epoch_49.pth", help='path to .pth')
parser.add_argument('--savepth', default="generate_data", help='generate_data')
parser.add_argument('--looptimes', default=40, help='generate_data num=looptime*20')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='checkpoints', help='folder to save checkpoints')
parser.add_argument('--imf', default='images', help='folder to save images')
parser.add_argument('--checkpoint_every', default=5, help='number of epochs after which saving checkpoints') 
parser.add_argument('--tensorboard_image_every', default=5, help='interval for displaying images on tensorboard') 
parser.add_argument('--model_type', default='cnn', help='model type')
opt = parser.parse_args()

#Create writer for tensorboard
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
run_name = f"{opt.run_tag}_{date}" if opt.run_tag != '' else date
log_dir_name = os.path.join(opt.logdir, run_name)
print(log_dir_name)
writer = SummaryWriter('log_dir_name')
writer.add_text('Options', str(opt), 0)
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
try:
    os.makedirs(opt.imf)
except OSError:
    pass

# 初始化
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("You have a cuda device, so you might want to run with --cuda as option")

############################dataset####################################################
if opt.dataset == "npz":
    dataset = npzDataset(opt.dataset_path, 4 ,4)

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
device = torch.device("cuda:0" if opt.cuda else "cpu")

if opt.model_type == "cnn":

    netM = cnn(input_size=1, output_size=1, padding = 2,kernel_size=3, stride=1,dropout=0).to(device) # change in_dim = 25

assert netM

if opt.netM != '':
    netG.load_state_dict(torch.load(opt.netM)) 

print("|CNN Architecture|\n", netM)