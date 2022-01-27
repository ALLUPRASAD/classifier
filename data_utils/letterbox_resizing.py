import argparse
import os
import glob
import numpy as np
import imutils
import cv2
from joblib import Parallel,delayed
from tqdm import tqdm
########### Help ###########
 
'''
#size  = (h,w)

python letterbox_resizing.py \
    --data_dir /Users/aman.gupta/Documents/self/datasets/blank_page/ \
    --size 224 224 \
    --output_dir /Users/aman.gupta/Documents/self/datasets/blank_page/letterbox_training_data 

'''

###########################

def convert_to_letterbox(path,size):
    
    img = cv2.imread(path)
    h,w = img.shape[:2]

    final_img = np.zeros((size[0],size[1],3))
    inter = cv2.INTER_CUBIC
    if w>h:
        if w>size[1]:
            inter = cv2.INTER_AREA
        resized_img = imutils.resize(img, width=size[1],inter = inter)
    else:
        if h > size[0]:
            inter = cv2.INTER_AREA
        resized_img = imutils.resize(img, height=size[0],inter = inter)

    new_h,new_w = resized_img.shape[:2]
    final_img[:new_h,:new_w,:] = resized_img
    return final_img

def letterbox_img(path,size,output_dir):
    img_name = os.path.basename(path)
    final_img=convert_to_letterbox(path,size)
    cv2.imwrite(os.path.join(output_dir,img_name),final_img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="this script resizes images to fit into specified dimentions without changing original spatial info")
    
    parser.add_argument("--data_dir",required = True,help="training data path")
    parser.add_argument("--size",default = [224,224],type = int,nargs='+', help="size of final image")
    parser.add_argument("--output_dir",required=False,default="./logs",type=str,help="dir to save logs")
    args = parser.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)

    folders =[path.split("/")[-1] for path in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir,path))]
    print("folders :",folders)

    for folder in folders:
        input_folder = os.path.join(args.data_dir,folder)
        output_folder = os.path.join(args.output_dir,folder)
        os.makedirs(output_folder,exist_ok=True)
        print(f"Processing {folder}")

        image_paths  = glob.glob(os.path.join(input_folder,"*.jpg"))
        print(f"Total files in dir :{len(image_paths)}")
        Parallel(n_jobs=-1,prefer="threads")(delayed(letterbox_img)(path,args.size,output_folder) for path in tqdm(image_paths))
        