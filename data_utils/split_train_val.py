import torch
from torchvision import datasets
import shutil
import argparse
import os
import numpy as np
from tqdm import tqdm
########### Help ###########
 
'''
#size  = (h,w)

python split_train_val.py \
    --data_dir /Users/aman.gupta/Documents/self/datasets/blank_page_detection/letterbox_training_data/ \
    --val_ratio 0.10 \
    --output_dir /Users/aman.gupta/Documents/self/datasets/blank_page_detection/splitted_letterbox_training_data 

'''

###########################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="this script splits classification data into train and val based on ratio provided by user")
    
    parser.add_argument("--data_dir",required = True,help="training data path")
    parser.add_argument("--val_ratio",default = 0.2,type = float, help="ratio of val in total data")
    parser.add_argument("--output_dir",required=False,default="./data/",type=str,help="dir to save images")
    args = parser.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)


    data = datasets.ImageFolder(args.data_dir)
    imgs_info = data.imgs
    classes = data.classes

    output_folders = ['train','val']
    for o_folder in output_folders:
        for folder in classes:
            os.makedirs(os.path.join(args.output_dir,o_folder,folder),exist_ok=True)

    print(f"Total classes:{len(classes)}")
    
    np.random.shuffle(imgs_info)
    
    num_samples = len(imgs_info)
    split = int(np.floor(args.val_ratio * num_samples))
    train_info, test_info = imgs_info[split:], imgs_info[:split]

    print(f"processing train...")

    for info in tqdm(train_info):
        folder = classes[info[1]]
        source = info[0]
        file_name = os.path.basename(source)
        destination = os.path.join(args.output_dir,output_folders[0],folder,file_name)
        shutil.copy(source, destination)
    
    print(f"processing val...")

    for info in tqdm(test_info):
        folder = classes[info[1]]
        source = info[0]
        file_name = os.path.basename(source)
        destination = os.path.join(args.output_dir,output_folders[1],folder,file_name)
        shutil.copy(source, destination)




    
