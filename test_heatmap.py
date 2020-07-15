import torch
import os
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from main import *
import skimage.io, skimage.transform
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from PIL import Image

def get_orig_position(liste):
    liste[:, 0] = ((liste[:, 0] + 0.5) * 48) - 0.5 + 24
    liste[:, 1] = ((liste[:, 1] + 0.5) * 48) - 0.5 + 24
    return liste


def plot_heatgraph(number_of_epochs, epochs, train_loss, precision, recall, f1_score, date_time):
    f, ax = plt.subplots(1, figsize = (10,5))
    plt.xlabel('epochs')
    plt.ylabel('train loss, precision, recall, F1 score')
    ax.plot(epochs, train_loss, 'r', label = 'Train loss')
    ax.plot(epochs, precision, 'g', label = 'Precision')
    ax.plot(epochs, recall, 'b', label = 'Recall')
    ax.plot(epochs, f1_score, 'm', label = 'F1 score')
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
    plt.savefig('../Plot/heatgraph_{}.svg'.format(date_time.strftime('%d-%m-%y_%H:%M')), dpi = 100)


def plot_heatmap(img_name, epoch, original_image, output_greater, list_label, list_max, date_time):
    width_origimg = original_image.shape[1]
    height_origimg = original_image.shape[0]
    print('{}x{}'.format(width_origimg, height_origimg))
    width_out = output_greater.shape[1]
    height_out = output_greater.shape[0]
    print('{}x{}'.format(width_out, height_out))

    extent = [0, width_origimg, height_origimg, 0]

    fig, ax = plt.subplots(figsize=(20,10))
    ax.imshow(original_image, cmap='gray', extent=extent)
    heatmap = ax.imshow(output_greater, interpolation='bicubic', alpha=0.4, extent=extent, cmap='viridis')
    trans_heatmap = mtransforms.Affine2D().scale((4000-24)/4000, (3000-24)/3000).translate(24,24) + ax.transData
    heatmap.set_transform(trans_heatmap)
    ax.axis('off')
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel(None, rotation=-90, va='bottom')
    plt.savefig('../Heatmap/{}_heatmap_overlay_{}_{}'.format(img_name, date_time.strftime('%d-%m-%y_%H:%M'), epoch), bbox_inches='tight')
    plt.close(fig)


def plot_coordinates(img_name, epoch, original_image, list_label, list_max, date_time):
    fig, ax =  plt.subplots(figsize=(20, 10))
    ax.imshow(original_image, cmap='gray')
    ax.scatter(list_label[:, 0], list_label[:, 1], marker='+')
    ax.scatter(list_max[:, 0], list_max[:, 1], marker='.')
    plt.axis('off')
    plt.savefig('../Heatmap/{}_label_max_coordinates_{}_{}'.format(img_name, date_time.strftime('%d-%m-%y_%H:%M'), epoch), bbox_inches='tight')
    plt.close(fig)


def create_heatmaps(img_name, epoch, original_image, output_greater, image_max, mask, list_label, list_max, net_coord, date_time):
    # heatmap overlaid onto raw image
    plot_heatmap(img_name, epoch, original_image, output_greater, list_label, list_max, date_time)

    # compare points of list_label to points of list_max
    plot_coordinates(img_name, epoch, original_image, list_label, list_max, date_time)


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
    mask &= output_greater > 0.50

    net_coord = np.nonzero(mask)
    net_coord = np.fliplr(np.column_stack(net_coord))

    list_max = get_orig_position(net_coord.copy())

    j = json.load(open(json_filename))
    list_label = j['positives']
    list_label.sort()
    list_label = np.array(list_label)
    
    print('*** list_max (maxima of net_coord) ***\n{}\n*** list_label (marked bees) ***\n{}'.format(list_max, list_label))

    if (list_max.size != 0 and epoch == number_of_epochs-1):
        create_heatmaps(img_name, epoch, original_image, output_greater, image_max, mask, list_label, list_max, net_coord, date_time)
    
    num_pred, true_positives, false_positives, false_negatives = create_score(img_name, epoch, list_label, list_max, num_pred, true_positives, false_positives, false_negatives)

    return num_pred, true_positives, false_positives, false_negatives
    

def loop_epochs(given_arguments, net, date_time, number_of_epochs):
    f1_score = []
    precision = []
    recall = []

    save_epoch = 0

    for epoch in range(number_of_epochs):
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


def heatmap(given_arguments, net, date_time, number_of_epochs, train_loss, epochs):
    precision, recall, f1_score = loop_epochs(given_arguments, net, date_time, number_of_epochs)
    plot_heatgraph(number_of_epochs, epochs, train_loss, precision, recall, f1_score, date_time)
