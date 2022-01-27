import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
from model import Model
import glob
import configs
import argparse
from PIL import Image
import os
import utils
import pandas as pd


########### Help ###########

"""
Predicts and saves outputs in csv file

python predict.py \
    --data_dir /Users/aman.gupta/Documents/self/datasets/Client_Files/version1_output_rectified/all_data/images/output/VAISHNVI/GHC07016074/ \
    --model_path ./logs/rotation_detection/checkpoints/exp0/best.pth \
    --output_dir ./results/exp0/last/VAISHNVI/GHC07016074/ \
    --csv_name GHC07016074.csv
######

"""

###########################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="this script do inference from the classification model"
    )

    parser.add_argument("--data_dir", required=True, help="test data path")
    parser.add_argument("--model_path", required=True, help="model path")
    parser.add_argument(
        "--output_dir",
        required=False,
        default="./logs",
        type=str,
        help="dir to save results",
    )
    parser.add_argument(
        "--csv_name",
        required=False,
        default="output.csv",
        type=str,
        help="output csv name",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ##load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_obj = Model(len(configs.CLASSES), device)
    model, optimizer = model_obj.model, model_obj.optimizer

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["train_loss"]

    model.eval()

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    softmax_fun = lambda x: np.exp(x) / sum(np.exp(x))

    image_paths = glob.glob(os.path.join(args.data_dir, "*.jpg"))
    print(f"total paths :{len(image_paths)}")

    fig = plt.figure(figsize=(10, 10))
    row = 5
    batch = 0
    num_imgs = 30000
    start = batch * num_imgs
    end = start + num_imgs

    csv_data = {"image_name": [], "class": [], "conf": []}
    for idx, img_path in tqdm(enumerate(image_paths[start:end])):
        img = utils.convert_to_letterbox(img_path,size = (224,224))
        # img = Image.open(img_path)
        # img = Image.fromarray(np.uint8(utils.letterbox_img(img)))
        image_tensor = test_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        output = model(image_tensor.to(device))

        output_val = np.round(softmax_fun(np.squeeze(output.data.cpu().numpy())), 2)
        index = output_val.argmax()

        csv_data["image_name"].append(os.path.basename(img_path))
        csv_data["class"].append(configs.CLASSES[index])
        csv_data["conf"].append(output_val[index])

        # sub = fig.add_subplot(5, 5, idx+1)
        # sub.set_title(str(configs.CLASSES[index]))
        # plt.axis('off')
        # plt.imshow(img)

    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(args.output_dir, args.csv_name), index=False)
    # plt.show()