import os
import json
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np

#all folder from 0-n




def avrg_plot(x):
    train_loss_avrg = []
    val_loss_avrg = []
    accuracy_avrg = []
    f1_avrg  = []
    image_count = []
    
    for a in range(x):
        image_count.append(sum([len(files) for r, d, files in os.walk("./ground_truths_"+ str(a) +"/training_images/0 notpollen")]) + sum([len(files) for r, d, files in os.walk("./ground_truths_"+ str(a) +"/test_images/0 notpollen")])) 
        data1 = json.load(open("./Plot/" + str(a) + "/test_data.json"))
        data2 = json.load(open("./Plot/" + str(a) + "/train_data.json"))
        train_loss_avrg.append(sum(data2['Train loss'])/ len(data2['Train loss']))
        val_loss_avrg.append(sum(data1['Validation loss'])/ len(data1['Validation loss']))
        accuracy_avrg.append(sum(data1['Accuracy'])/ len(data1['Accuracy']))
        f1_avrg.append(sum(data1['F1-Score'])/ len(data1['F1-Score']))
    
    f, ax = plt.subplots(1, figsize = (10,5))
    plt.xlabel('sum negativ examples used')
    plt.ylabel('avrg. train/validation loss/test accuracy/F1 score')
    ax.plot(image_count, train_loss_avrg, 'r', label = ' avrg Train loss')
    ax.plot(image_count, val_loss_avrg, 'b', label = 'avrg Validation loss')
    ax.plot(image_count, accuracy_avrg, 'g', label = 'avrg (Test) Accuracy / 100')
    ax.plot(image_count, f1_avrg, 'm', label = 'avrg F1 Score')
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    leg = ax.legend(bbox_to_anchor=(1.01, 0.5), loc = 'center left', frameon = True)
    leg.get_frame().set_edgecolor('k')
    plt.grid(False)
    plt.tight_layout()
    axes = plt.gca()
    axes.set_xlim([min(image_count),max(image_count)])
    axes.set_ylim([0,1])


def plot_all(x):
    image_count = (sum([len(files) for r, d, files in os.walk("./ground_truths_"+ str(x) +"/training_images/0 notpollen")])) + (sum([len(files) for r, d, files in os.walk("./ground_truths_"+ str(x) +"/test_images/0 notpollen")]))
    data1 = json.load(open("./Plot/" + str(x) + "/test_data.json"))
    data2 = json.load(open("./Plot/" + str(x) + "/train_data.json"))
    epochs = (data1['Epoch'])
    train_loss = (data2['Train loss'])
    val_loss = (data1['Validation loss'])
    accuracy = (data1['Accuracy'])
    f1 = (data1['F1-Score'])
    
    return len(epochs), epochs,train_loss,accuracy,f1,val_loss, image_count
    
def plot_train_test(number_of_epochs, epochs, train_loss, test_accuracy, f1_score, validation_loss, x):
    f, ax = plt.subplots(1, figsize = (10,5))
    plt.title('with ' + str(x) + ' negative examples')
    plt.xlabel('epochs')
    plt.ylabel('train loss, validation loss, test accuracy, F1 score')
    ax.plot(epochs, train_loss, 'r', label = 'Train loss')
    ax.plot(epochs, validation_loss, 'b', label = 'Validation loss')
    ax.plot(epochs, test_accuracy, 'g', label = '(Test) Accuracy / 100')
    ax.plot(epochs, f1_score, 'm', label = 'F1 Score')
    plt.axis([0, (number_of_epochs - 1), 0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, number_of_epochs, (number_of_epochs // 10)))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    leg = ax.legend(bbox_to_anchor=(1.01, 0.5), loc = 'center left', frameon = True)
    leg.get_frame().set_edgecolor('k')
    plt.grid(False)
    plt.tight_layout()

def main():
    v = len(next(os.walk('Plot/'))[1])
    avrg_plot(v)
    for a in range(v):
        number_of_epochs, epochs, train_loss, test_accuracy, f1_score, validation_loss, x = plot_all(a)
        plot_train_test(number_of_epochs, epochs, train_loss, test_accuracy, f1_score, validation_loss, x)   
    
if __name__ == '__main__':
    main()
