import sys
sys.path.append('./')
sys.path.append('../')

import time
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import os
import argparse

import src.models.resnet_cifar as resnet
from src.utils.train_functions import data_iteration
from src.utils.utils import make_output_folders, get_logger
from src.dataset.cifar import get_dataset

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#             'dog', 'frog', 'horse', 'ship', 'truck')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--arch', default='resnet20', type=str, help='path for storing exp output and model weights')
parser.add_argument('--data_path', default='../../../data/', type=str)
parser.add_argument('--dataset', default='cifar10', type=str, help='select dataset to use')
parser.add_argument('--save_root', default='./runs/', type=str, help='path for storing exp output and model weights')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=200, type=int, help='learning epochs')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
parser.add_argument('--scheduler', default='cosine', choices=['step', 'cosine', 'none'])
parser.add_argument('--augmentation', default=0, type=int, choices=[0, 1, 2])
args = parser.parse_args()


# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def solution():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    
    args=parser.parse_args()
    make_output_folders(args)
    logger = get_logger(args)
    
    start = time.time()
    
    dataloaders = get_dataset(args)
    
    model = getattr(resnet, args.arch)(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    
    for epoch in range(args.epochs):
        # train_acc, train_loss = train(dataloaders['train'], model, optimizer, criterion, epoch, device)
        # val_acc, val_loss = test(dataloaders['val'], model, criterion, device)
        train_acc, train_loss = data_iteration(dataloaders['train'], model, criterion, device, optimizer, phase='train')
        val_acc, val_loss = data_iteration(dataloaders['val'], model, criterion, device, optimizer, phase='val')
        
        scheduler.step()
                
        if val_acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            
            torch.save(state, os.path.join(args.save_root, 'best_model.pth'))
            best_acc = val_acc
            
        logger.info('Epoch: %d | Train Pec@1: %.3f%% loss: %.3f\n' % (epoch, train_acc, train_loss))
        logger.info('Epoch: %d | Best Prec@1: %.3f%% | Prec@1: %.3f%% loss: %.3f\n' % (epoch, best_acc, val_acc, val_loss))
        
    logger.info('Total Training time: %.2f' %(time.time()-start) )
    
    return best_acc

if __name__ == '__main__':
    solution()