import json
import torch
from torch.utils import data
import numpy as np


def store_json_training(epochs, train_loss):
    d = {
        "Train loss" : train_loss,
        "Epoch" : epochs
    }
    f = open('./Plot/train_data.json', 'w')
    json.dump(d, f, indent = 4)


def train_net(trainloader, net, criterion, optimizer, epoch, running_loss, t_loss, train_i, train_loss, number_of_epochs):
    save_epochs = open('./Training/training_epoch-{}'.format(epoch), 'a')
    save_epochs.close()

    for index, batch in enumerate(trainloader, 0):
        inputs, labels = batch

        optimizer.zero_grad()
        outputs = net(inputs)
        labels = labels.float().reshape(torch.Size(outputs.shape))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss = float(loss.data.cpu().numpy())
        running_loss += loss
        t_loss += loss
        train_i = index + 1

        if index % 10 == 9:
            print('Train Epoch: %d\t Index: %3d\t Loss %.6f' %
                    (epoch + 1, index + 1, running_loss / 10))
            running_loss = 0.0
    print(train_i)
        
    train_loss += [t_loss / train_i]

    print('Train Epoch: %d\t Average loss: %.6f' %
                    (epoch + 1, t_loss / train_i))

    torch.save(net.state_dict(), './Training/training_epoch-{}'.format(epoch))

    print('*** Finished Training - Epoch {} ***'.format(epoch + 1))


def loop_epochs(trainloader, net, criterion, optimizer, number_of_epochs):
 
    epochs = []
    train_loss = []

    for epoch in range(number_of_epochs):
        running_loss = 0.0
        t_loss = 0.0
        train_i = 0

        epochs += [epoch]

        print('*** Start Training - Epoch {} ***'.format(epoch + 1))
        net.train()
        train_net(trainloader, net, criterion, optimizer, epoch, running_loss, t_loss, train_i, train_loss, number_of_epochs)    

    return net, epochs, train_loss


def train(trainloader, net, criterion, optimizer, number_of_epochs, date_time):
    net_trained, epochs, train_loss = loop_epochs(trainloader, net, criterion, optimizer, number_of_epochs)
    store_json_training(epochs, train_loss)