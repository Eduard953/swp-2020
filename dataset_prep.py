
"""
# folder structure 


Software Projekt/
│
├── BeesBook/
│   │
│   ├── 0/
│   │     ├── ann/
│   │     │    └── jsonfiles...
│   │     └── img/
│   │          └── images...
│   ├── 1/
│         ├── ann/
│         │    └── jsonfiles...
│         └── img/
│              └── images...
│    
│        # 3, 4 etc. for each batch of annotated imgs from supervisly
│
├── ground_truths/
│    │
│    ├──training_images/
│    │    │
│    │    ├── 0 notpollen/
│    │    │
│    │    └── 1 pollen/
│    │    
│    └───test_images/
│         │
│         ├── 0 notpollen/
│         │
│         └── 1 pollen/
│
└── dataset_prep.py



also example img:
GT_1_img_1_67_0.png
GT_1 -> ground truth 1 (can be 0)
img_1 -> img from folder 1
_67 -> 67th image or 134th frame from that folder
_0 -> iteration from the same image
"""

import os
import json
from PIL import Image
import random
from random import sample
from distutils.dir_util import copy_tree


# paths to respective directory for positve and negative examples 

dir_target_training_0 = "ground_truths/training_images/0 notpollen/"
dir_target_training_1 = "ground_truths/training_images/1 pollen/"
dir_target_test_0 = "ground_truths/test_images/0 notpollen/"
dir_target_test_1 = "ground_truths/test_images/1 pollen/"
    

# data preperation function takes 3 inputs
# image_size requires the the resolution of images used
# crop_size is the output size of the cropped images for the CNN
# number_of_random_crops is the amount of negative cropped images that do not include the target
    
def data_prep(image_size, crop_size, number_of_random_crops):
    crop_size = crop_size // 2
    img_shape = image_size
    
    
    # move through directory of BeesBook/
    
    for x in range(len(next(os.walk('BeesBook/'))[1])):
        
        dir_json = "BeesBook/" + str(x) + "/ann/"
        dir_img = "BeesBook/" + str(x) + "/img/"
        list_json = os.listdir(dir_json)
        list_img = os.listdir(dir_img)
    
        #lists the coordinates of the target from their respective JSON file
        
        for b in range(len(list_img)):
    
            data = json.load(open(dir_json + list_json[b],))
            data_len = len(data['objects'])
            coords = []
    
            if data_len > 0:
                for a in range(data_len):
                    
                    for i in data['objects'][a]['points']['exterior']: 
                        coords.append(i)
                
                    # crops the target image and saves it after
                
                    current_img = Image.open(dir_img + list_img[b])
                    x1 = coords[a][0] - crop_size
                    y1 = coords[a][1] - crop_size
                    x2 = coords[a][0] + crop_size
                    y2 = coords[a][1] + crop_size
                    current_img = current_img.crop((x1,y1,x2,y2))
                    current_img.save( dir_target_training_1 + "GT_1_img_" + str(x) + '_' + str(b) + '_' + str(a) + ".png")
             
            elif data_len == 0:
                for t in range(number_of_random_crops):
                    
                    # randomly crops image with no target
                    
                    current_img = Image.open(dir_img + list_img[b])
                    rnd_coord = (random.randint(crop_size,img_shape[0]-crop_size),random.randint(crop_size,img_shape[1]-crop_size))
                    x1 = rnd_coord[0] - crop_size
                    y1 = rnd_coord[1] - crop_size
                    x2 = rnd_coord[0] + crop_size
                    y2 = rnd_coord[1] + crop_size
                    current_img = current_img.crop((x1,y1,x2,y2))
                    current_img.save( dir_target_training_0 + "GT_0_img_" + str(x) + '_' + str(b) + '_' + str(t) + ".png")
                    
        print("folder " + str(x + 1) + "/" + str(len(next(os.walk('BeesBook/'))[1]))+ " done!")
            
        
#split the dataset into test and training directories for use with a specific ratio (20% optimal)

def data_split(ratio):
    c = next(os.walk(dir_target_training_0))[2]
    c_o_dot_len = int(ratio * len(c))
        
    for y in range(c_o_dot_len):
        
        c = next(os.walk(dir_target_training_0))[2]
        c_len = len(c)
    
        rnd = random.randint(0,c_len -1)
        os.rename(dir_target_training_0 + str(c[rnd]), dir_target_test_0 + str(c[rnd]))
        
    print("first split done!")
    
    l = next(os.walk(dir_target_training_1))[2]
    l_o_dot_len = int(ratio * len(l)) 
    
    for o in range(l_o_dot_len):
        
        l = next(os.walk(dir_target_training_1))[2]
        l_len = len(l)
    
        rnd = random.randint(0,l_len -1)
        os.rename(dir_target_training_1 + str(l[rnd]), dir_target_test_1 + str(l[rnd]))
    
        
    print("second split done!")
    
# scales down negaive examples each iteration    
    
def data_scale(ratio, iterations):
    gtlist = ["./ground_truths/", "./ground_truths_1/"  ,"./ground_truths_2/" ,"./ground_truths_3/" ,"./ground_truths_4/" ,"./ground_truths_5/","./ground_truths_6/","./ground_truths_7/","./ground_truths_8/","./ground_truths_9/" ,"./ground_truths_10/","./ground_truths_11/","./ground_truths_12/","./ground_truths_13/","./ground_truths_14/","./ground_truths_15/" ]
    for runs in range(iterations):
        copy_tree(gtlist[runs], gtlist[runs+1])
        files = os.listdir(gtlist[runs+1] + "training_images/0 notpollen/")
        len_files = int(ratio * len(files))
        for file in sample(files,len_files):
            os.remove(gtlist[runs+1] + "training_images/0 notpollen/" + str(file))
        files = os.listdir(gtlist[runs+1] + "test_images/0 notpollen/")
        len_files = int(ratio * len(files))
        for file in sample(files,len_files):
            os.remove(gtlist[runs+1] + "test_images/0 notpollen/" + str(file))
        print("copy " + str(runs+1) +"/"+ str(iterations))
    
    
def main():
    data_prep(image_size=(4000,3000),crop_size=84, number_of_random_crops=10)
    data_split(ratio=0.2)
    data_scale(ratio=0.2, iterations=15)
    
if __name__ == '__main__':
    main()
