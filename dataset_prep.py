
"""
# folder structure 


Software Projekt/
├── BeesBook/
│   ├── 1/
│   │     ├── ann/
│   │     │    └── jsonfiles...
│   │     └── img/
│   │          └── images...
│   ├── 2/
│         ├── ann/
│         │    └── jsonfiles...
│         └── img/
│              └── images...
    
        # 3, 4 etc. for each batch of annotated imgs from supervisly

├── ground_truths/
└── dataset_crop.py

"""



#imports lol

import os
import json
from PIL import Image
import random
import csv


# global vars

crop_size = 84 / 2
number_of_random_crops = 10
img_shape = (4000,3000)
dir_target = "ground_truths/"


#loop for each folder with annotated images 

for x in range(len(next(os.walk('BeesBook/'))[1])):
    
    
    #write a csv file

    with open('ground_truths/Aforagers.csv', 'w', newline='') as file:
        writer = csv.writer(file)
    
        # folder dir, edit for your pc?

        dir_json = "BeesBook/" + str(x+1) + "/ann/"
        dir_img = "BeesBook/" + str(x+1) + "/img/"


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
                    current_img.save( dir_target + "GT_1_img_" + str(x+1) + '_' + str(b) + '_' + str(a) + ".png")
                
                
                    #write into csv file with ,1
                
                    writer.writerow(["GT_1_img_" + str(x+1) + '_' + str(b) + '_' + str(a) + ".png", 1])
         
    
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
                    current_img.save( dir_target + "GT_0_img_" + str(x+1) + '_' + str(b) + '_' + str(a) + ".png")
                    
                    #write into csv file with ,0
                    
                    writer.writerow(["GT_0_img_" + str(x+1) + '_' + str(b) + '_' + str(a) + ".png", 0])
            