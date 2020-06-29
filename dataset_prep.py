
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



#imports lol

import os
import json
from PIL import Image
import random

# global vars

crop_size = 84 / 2
number_of_random_crops = 10
img_shape = (4000,3000)
dir_target_training_0 = "ground_truths/training_images/0 notpollen/"
dir_target_training_1 = "ground_truths/training_images/1 pollen/"
dir_target_test_0 = "ground_truths/test_images/0 notpollen/"
dir_target_test_1 = "ground_truths/test_images/1 pollen/"


#loop for each folder with annotated images 

for x in range(len(next(os.walk('BeesBook/'))[1])):
    

    # folder dir, edit for your pc?

    dir_json = "BeesBook/" + str(x) + "/ann/"
    dir_img = "BeesBook/" + str(x) + "/img/"


    # list of json/img

    list_json = os.listdir(dir_json)
    list_img = os.listdir(dir_img)
  
    #loop through folder
      
    for b in range(len(list_img)):

    
        # load json file
    
        data = json.load(open(dir_json + list_json[b],))

    
        # number of marked points
        
        data_len = len(data['objects'])
        coords = []


        # if points in image, crop it, save it for GT1

    
        if data_len > 0:
    
            for a in range(data_len):
                for i in data['objects'][a]['points']['exterior']: 
                    coords.append(i)
            
            
                # select current image

                current_img = Image.open(dir_img + list_img[b])
    
                x1 = coords[a][0] - crop_size
                y1 = coords[a][1] - crop_size
                x2 = coords[a][0] + crop_size
                y2 = coords[a][1] + crop_size
        
                
                #save img
                
                current_img = current_img.crop((x1,y1,x2,y2))
                current_img.save( dir_target_training_1 + "GT_1_img_" + str(x) + '_' + str(b) + '_' + str(a) + ".png")
         
    
            # if no points in image, random crops, save it for GT0
        
        elif data_len == 0:
        
            
            for t in range(number_of_random_crops):
                
                            
                # select current image
    
                current_img = Image.open(dir_img + list_img[b])
        
        
                rnd_coord = (random.randint(crop_size,img_shape[0]-crop_size),random.randint(crop_size,img_shape[1]-crop_size))
                
                x1 = rnd_coord[0] - crop_size
                y1 = rnd_coord[1] - crop_size
                x2 = rnd_coord[0] + crop_size
                y2 = rnd_coord[1] + crop_size
        
        
                #save img
        
                current_img = current_img.crop((x1,y1,x2,y2))
                current_img.save( dir_target_training_0 + "GT_0_img_" + str(x) + '_' + str(b) + '_' + str(t) + ".png")
                
    print("folder done!")
            
#spit test and training

c = next(os.walk(dir_target_training_0))[2]
c_o_dot_len = int(0.2 * len(c))
    
for y in range(c_o_dot_len):
    
    c = next(os.walk(dir_target_training_0))[2]
    c_len = len(c)
    c_dot_len = int(0.2 * len(c))

    rnd = random.randint(0,c_len -1)
    os.rename(dir_target_training_0 + str(c[rnd]), dir_target_test_0 + str(c[rnd]))
    
print("first split done!")

l = next(os.walk(dir_target_training_1))[2]
l_o_dot_len = int(0.2 * len(l)) 

for o in range(l_o_dot_len):
    
    l = next(os.walk(dir_target_training_1))[2]
    l_len = len(l)
    l_dot_len = int(0.2 * len(l))

    rnd = random.randint(0,l_len -1)
    os.rename(dir_target_training_1 + str(l[rnd]), dir_target_test_1 + str(l[rnd]))
    
print("second split done!")
