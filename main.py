#!/usr/bin/env python3

"""
# folder structure
├── main.py
├── train_network.py
├── test_network.py
├── ground_truths/
│   ├── training_images/
│   │     ├── 0 notpollen/
│   │     │    └── GT_0-images (80%)
│   │     └── 1 pollen/
│   │          └── GT_1-images(80%)
│   ├── test_images/
│         ├── 0 notpollen/
│         │    └── GT_0-images (20%)
│         └── 1 pollen/
│              └── GT_1-images (20%)
"""

import json, sys
import datetime
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import train_network, test_network


def load_data():
    b_size = 100

    transform_train = transforms.Compose([
        transforms.RandomOrder([
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5)
        ]),
        transforms.CenterCrop(54),
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(48),
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.ImageFolder(root='./ground_truths/training_images', transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root='./ground_truths/test_images', transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = b_size, num_workers=2, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset)

    print('class_to_idx trainset: {}\t class_to_idx testset: {}'.format(trainset.class_to_idx, testset.class_to_idx))

    classes = ('notpollen', 'pollen')

    return trainloader, testloader


class Net(nn.Module):
    """Define a Convolutional Neural Network:
       1@32x32 -> 30@16x16 -> 60@8x8 -> 60@4x4 -> 120@2x2 -> 120@1x1 -> 1@1x1"""
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(1, 30, 5, padding=2)
        self.conv2 = nn.Conv2d(30, 60, 3, padding=1)
        self.conv3 = nn.Conv2d(60, 60, 3, padding=1)
        self.conv4 = nn.Conv2d(60, 120, 3, padding=1)
        self.conv5 = nn.Conv2d(120, 120, 3, padding=1)
        self.conv6 = nn.Conv2d(120, 1, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = torch.sigmoid(self.conv6(x))
        
        return x


def main(args):
    trainloader, testloader = load_data()
    net = Net()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())
    date_time = datetime.datetime.now()
    number_of_epochs = 10
    
    if len(args) < 2:
        print('usage: main.py (train|test)')
    
    elif args[1] == 'train':
        train_network.train(trainloader, net, criterion, optimizer, number_of_epochs, date_time)

    elif args[1] == 'test':
        j = json.load(open('./Plot/train_data.json'))
        train_loss = j["Train loss"]
        epochs = j["Epoch"]
        test_network.test(testloader, net, criterion, number_of_epochs, date_time, train_loss, epochs)
        
    else:
        print('unknown command: {}'.format(args[1]))

if __name__ == '__main__':
    main(sys.argv)
