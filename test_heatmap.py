import torch
import os
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from main import *
from torchvision import transforms
import skimage.io, skimage.transform
from skimage.feature import peak_local_max
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from scipy.spatial.kdtree import KDTree
import plot

def get_orig_position(liste):
    liste[:, 0] = (liste[:, 0] *48) + 24
    liste[:, 1] = (liste[:, 1] * 48) + 24
    return liste


def create_score(img_name, epoch, list_label, list_max, num_pred, true_positives, false_positives, false_negatives):
    i = 0
    if (list_max.size == 0):
        prec = 0.0
        rec = 0.0
        f1 = 0.0
    else:
        while i < len(list_label):
            target = list_label[i]
            print(target)
            max_distance = 24
            distances = np.linalg.norm(list_max-target, axis=1)
            distances[distances > max_distance] = np.inf
            best_match = np.argmin(distances)
            match_distance = distances[best_match]
            if np.isinf(match_distance):
                false_negatives += 1
            else:
                true_positives += 1
                list_max[best_match] = np.inf
            print('best_match: {}\t match_distance: {}'.format(best_match, match_distance))
            i = i + 1
        
        num_pred += len(list_max)
        false_positives = num_pred - true_positives
    
    return num_pred, true_positives, false_positives, false_negatives


def create_coord_lists(number_of_epochs, json_filename, image_filename, img_name, net, epoch, num_pred, true_positives, false_positives, false_negatives, date_time):
    im = skimage.io.imread(image_filename)
    if len(im.shape) == 3:
        im = rgb2gray(im)
    else:
        im = im /255.0
    im = im[:,:]

    original_image = im.copy()
    print('im.shape: {}\nim.dtype: {}\n im.max: {}'.format(im.shape, im.dtype, im.max()))
    im = skimage.transform.rescale(im, scale=(32 / 48))
    im = im.reshape((1, 1, im.shape[0], im.shape[1]))
    im = im.astype(np.float32)
    net.eval()
    print('im.shape: ', im.shape)
    with torch.no_grad():
        output = net(torch.tensor(im)) #.cuda())
        predicted = output > 0.50
        print('output.shape: ', output.shape)
        print('output: {}\t predicted: {}'.format(output, predicted))

    output = output.data.cpu().numpy()
    output_greater = output[0, 0].copy()

    image_max = ndi.maximum_filter(output_greater.copy(), size=8, mode='constant')
    mask = output_greater == image_max
    mask &= output_greater > 0.96

    net_coord = np.nonzero(mask)
    net_coord = np.fliplr(np.column_stack(net_coord))

    list_max = get_orig_position(net_coord.copy())

    j = json.load(open(json_filename))
    list_label = j['positives']
    list_label.sort()
    list_label = np.array(list_label)
    
    print('*** list_max (maxima of net_coord) ***\n{}\n*** list_label (marked bees) ***\n{}'.format(list_max, list_label))

    num_pred, true_positives, false_positives, false_negatives = create_score(img_name, epoch, list_label, list_max, num_pred, true_positives, false_positives, false_negatives)

    return num_pred, true_positives, false_positives, false_negatives
    

def loop_epochs(given_arguments, net, date_time, number_of_epochs):
    f1_score = []
    precision = []
    recall = []

    save_epoch = 0

    for epoch in range(number_of_epochs-1,number_of_epochs):
        save_epoch = epoch
        print('*** Test on full images using state of epoch {} ***'.format(epoch))
        num_pred = 0.0
        true_positives = 0.0
        false_positives = 0.0
        false_negatives = 0.0

        for json_filename in given_arguments:
            j = json.load(open(json_filename))
            dirname_json, filename = os.path.split(json_filename)
            image_filename = dirname_json.replace('new_ann', 'img/') + os.path.splitext(filename)[0] + '.png'
            dirname, img_name = os.path.split(image_filename.replace('.png', ''))
            print(img_name)
        
            net.load_state_dict(torch.load('../Training/training_epoch-{}'.format(save_epoch)))
            num_pred, true_positives, false_positives, false_negatives = create_coord_lists(number_of_epochs, json_filename, image_filename, img_name, net, epoch, num_pred, true_positives, false_positives, false_negatives, date_time)
            
        if (false_positives == 0 and true_positives == 0):
            prec = 0
        else:
            prec = true_positives / (true_positives + false_positives)
        if (false_negatives == 0 and true_positives == 0):
            rec = 0
        else:
            rec = true_positives / (true_positives + false_negatives)
        numerator = prec * rec
        denominator = prec + rec
        if (denominator == 0):
            f1 = 0
        else:
            f1 = 2 * (numerator / denominator)
 
        precision += [prec]
        recall += [rec]
        f1_score += [f1]
        print('Precision:', prec)
        print('Recall:', rec)
        print('F1_Score:', f1)
        if epoch == (number_of_epochs-1):
            print('Test Epoch: %d\n#predicted: {}\n#true positives: {}\n#false positives: {}\n#false negatives: {}'.format(num_pred, true_positives, false_positives, false_negatives) % (epoch+1))

    print('F1_Score: ', f1_score)

    return precision, recall, f1_score


def heatmap(given_arguments, net, date_time, number_of_epochs):
    precision, recall, f1_score = loop_epochs(given_arguments, net, date_time, number_of_epochs)