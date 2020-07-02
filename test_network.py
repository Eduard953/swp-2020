import json
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np

def store_json_test(epochs, test_accuracy, f1_score, validation_loss):
    d = {
        "Validation loss" : validation_loss,
        "Epoch" : epochs,
        "Accuracy" : test_accuracy,
        "F1-Score" : f1_score
    }
    f = open('./Plot/test_data.json', 'w')
    json.dump(d, f, indent = 4)


def plot_train_test(number_of_epochs, date_time, epochs, train_loss, test_accuracy, f1_score, validation_loss):
    f, ax = plt.subplots(1, figsize = (10,5))
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
    plt.savefig('./Plot/train_test_{}.svg'.format(date_time.strftime('%d-%m-%y_%H:%M')), dpi = 100)


def test_net(testloader, net, criterion, number_of_epochs, epoch, true_positives, true_negatives, false_positives, false_negatives, validation_loss, test_accuracy, f1_score):
    with torch.no_grad():
        val_loss = 0.0
        val_i = 0
        image_count = [6,3]

        for index, sample in enumerate(testloader, 0):
            inputs, labels = sample
            outputs = net(inputs)
            predicted = outputs > 0.90
            
            labels = labels.float().reshape(torch.Size(outputs.shape))

            loss = criterion(outputs, labels)
            
            loss = float(loss.data.cpu().numpy())
            val_loss += loss
            val_i = index + 1

            true_positives += int(predicted) and int(labels)
            true_negatives += not(int(predicted)) and not(int(labels))
            false_positives += int(predicted) and not(int(labels))
            false_negatives += not(int(predicted)) and int(labels)

            total = true_positives + true_negatives + false_positives + false_negatives
            correct = true_positives + true_negatives

            print('Predicted: {}\t Solution: {}'.format(int(predicted), int(labels)))
            print('output: %.6f' % float(outputs))
               
        validation_loss += [val_loss / val_i]

        if (false_positives == 0 and true_positives == 0):
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)
        if (false_negatives == 0 and true_positives == 0):
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)
        numerator = precision * recall
        denominator = precision + recall
        if (denominator == 0):
            f1 = 0
        else:
            f1 = 2 * (numerator / denominator)

        print('Test Epoch: %d\t Average validation loss: %.6f\n Precision: {}\t Recall: {}'.format(precision, recall) %
                    (epoch+1, val_loss/val_i))
    
        if epoch == (number_of_epochs-1):
            print('Test Epoch: %d\n #true positives: {}\n #true negatives: {}\n #false positives: {}\n #false negatives: {}'.format(true_positives, true_negatives, false_positives, false_negatives) % (epoch+1))

    test_accuracy += [correct / total]
    f1_score += [f1]

    print('*** Finished testing in epoch {} ***\n The current accuracy of the network on the test images is %d %%'.format(epoch + 1) % (100 * correct / total))


def loop_epochs(testloader, net, criterion, number_of_epochs):
    test_accuracy = []
    f1_score = []
    validation_loss = []

    for epoch in range(number_of_epochs):
        print('*** Start testing in epoch {} ***'.format(epoch + 1))
        true_positives = 0.0
        true_negatives = 0.0
        false_positives = 0.0
        false_negatives = 0.0

        net.load_state_dict(torch.load('./Training/training_epoch-{}'.format(epoch)))
        net.eval()
        test_net(testloader, net, criterion, number_of_epochs, epoch, true_positives, true_negatives, false_positives, false_negatives, validation_loss, test_accuracy, f1_score)

    return test_accuracy, f1_score, validation_loss


def test(testloader, net, criterion, number_of_epochs, date_time, train_loss, epochs):
    test_accuracy, f1_score, validation_loss = loop_epochs(testloader, net, criterion, number_of_epochs)
    store_json_test(epochs, test_accuracy, f1_score, validation_loss)
    plot_train_test(number_of_epochs, date_time, epochs, train_loss, test_accuracy, f1_score, validation_loss)
