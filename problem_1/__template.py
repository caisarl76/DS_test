import sys
sys.path.append('./')
sys.path.append('../')

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms

import argparse

import src.models.resnet_cifar as resnet
from src.utils.train_functions import *
from src.utils.utils import make_output_folders
from src.dataset.cifar import get_dataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data_path', default='../../../data', type=str, help='dataset path')
parser.add_argument('--subset_ratio', default=1.0, type=float)
parser.add_argument('--dataset', default='cifar10', type=str, help='select dataset to use')
parser.add_argument('--save_root', default='./runs/', type=str, help='path for storing exp output and model weights')
parser.add_argument('--epochs', default=50, type=int, help='learning epochs')
parser.add_argument('--batch_size', default=128, type=int)
args = parser.parse_args()

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def template():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args=parser.parse_args()
    
    dataloaders = get_dataset(args, transform=transform_train)
    
    model = resnet.resnet20().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    for _ in range(args.epochs):
        # start train iteration
        data_iteration(dataloaders['train'], model, criterion, device, optimizer, phase='train')
        # start val iteration
        data_iteration(dataloaders['val'], model, criterion, device, optimizer, phase='val')
    
    return model
    
if __name__ == '__main__':
    template()