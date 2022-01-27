import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
import configs
import time
import os
import torch.nn.functional as F
import cv2
import imutils
from PIL import Image
import itertools


def imbalanced_class_weights(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def load_split_train_test(data_dir, batch_size=64):
    size = (512, 512)
    # size = (224, 224)
    train_transforms = transforms.Compose(
        [  # transforms.RandomRotation(30),  # data augmentations are great
            # transforms.CenterCrop(768),  # but not in this case of map tiles
            #                                        transforms.RandomHorizontalFlip(),
            transforms.Resize(size),
            transforms.Grayscale(),  # new
            transforms.ToTensor(),
            # transforms.Normalize(
            #     [0.485, 0.456, 0.406],  # PyTorch recommends these but in this
            #     [0.229, 0.224, 0.225],
            # ),  # case I didn't get good results
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    test_transforms = transforms.Compose(
        [
            # transforms.CenterCrop(768),
            transforms.Resize(size),
            transforms.Grayscale(),  # new
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    train_data = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transforms
    )
    val_data = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=test_transforms
    )

    print(f"Total Train images :", len(train_data.imgs))
    # For unbalanced dataset we create a weighted sampler
    train_weights = imbalanced_class_weights(train_data.imgs, len(train_data.classes))
    train_weights = torch.DoubleTensor(train_weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        train_weights, len(train_weights)
    )

    print(f"Total Test images :", len(val_data.imgs))
    val_weights = imbalanced_class_weights(val_data.imgs, len(val_data.classes))
    val_weights = torch.DoubleTensor(val_weights)
    val_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        val_weights, len(val_weights)
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # labels = test_data.imgs
    # print("here :",len(labels[:]))
    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(valid_size * num_train))
    # np.random.shuffle(indices)
    # # from torch.utils.data.sampler import SubsetRandomSampler
    # # train_idx, test_idx = indices[split:], indices[:split]
    # # train_sampler = SubsetRandomSampler(train_idx)
    # # test_sampler = SubsetRandomSampler(test_idx)

    # weights = 1 / torch.Tensor(class_sample_count)
    # weights = weights.double()
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
    # trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return train_loader, val_loader


# helper functions


def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                configs.CLASSES[preds[idx]],
                probs[idx] * 100.0,
                configs.CLASSES[labels[idx]],
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
    return fig


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="gray")
    else:
        npimg = npimg.astype(int)
        plt.imshow(np.transpose(np.clip(npimg, 0, 255), (1, 2, 0)))


def save_checkpoint(path, file_name, epoch, model, optimizer, loss):

    os.makedirs(path, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": loss,
        },
        os.path.join(path, f"{file_name}.pth"),
    )


def letterbox_img(img, size=(224, 224)):
    img = np.array(img)
    h, w = img.shape[:2]

    final_img = np.ones((size[0], size[1], 3)) * 255
    inter = cv2.INTER_CUBIC
    if w > h:
        if w > size[1]:
            inter = cv2.INTER_AREA
        resized_img = imutils.resize(img, width=size[1], inter=inter)
    else:
        if h > size[0]:
            inter = cv2.INTER_AREA
        resized_img = imutils.resize(img, height=size[0], inter=inter)

    new_h, new_w = resized_img.shape[:2]
    final_img[:new_h, :new_w, :] = resized_img
    return final_img


def convert_to_letterbox(path, size):

    # img = cv2.imread(path)
    # h,w,c = img.shape
    img = Image.open(path)
    # print(img.size)
    w, h = img.size

    # inter = cv2.INTER_CUBIC
    if w > h:
        # if w>size[1]:
        #     inter = cv2.INTER_AREA
        # resized_img = imutils.resize(img, width=size[1],inter = inter)

        final_img = np.ones((w, w, 3), np.uint8) * 255
        # final_img = PIL.Image.new(mode = "RGB", size = (w, w), color = (255, 255, 255) )

    else:
        # if h > size[0]:
        #     inter = cv2.INTER_AREA
        # resized_img = imutils.resize(img, height=size[0],inter = inter)
        final_img = np.ones((h, h, 3), np.uint8) * 255
        # final_img = PIL.Image.new(mode = "RGB", size = (h, h), color = (255, 255, 255) )

    # print(img.size)

    final_img[:h, :w] = img
    final_img = Image.fromarray(final_img)
    final_img = final_img.resize((size[0], size[1]), Image.ANTIALIAS)
    # new_h,new_w = resized_img.shape[:2]
    # final_img[:new_h,:new_w,:] = resized_img

    return final_img


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"]
