#
# train.py
# @author amangupta0044@gmail.com
# @description
# @created 2020-12-09T16:35:56.524Z+05:30
# @last-modified Tuesday, 9th February 2021
#

########### Help ###########
"""
Step 1 : Modify data classes in config.py
Step 2 : Check model.py fo model type and final layer. (Current input 224x224). 
Step 3 : Run the training script.

python train.py \
    --data_dir ../training_data/org_training_data/splitted_training/ \
    --log_dir ./logs/ \
    --epochs 30 \
    --save_interval 1000 \
    --print_interval 50 \
    --batch_size 128 \
    --name exp8 \
    --lr 0.0001 \
    --checkpoint_path /home/ubuntu/aman/rotation_detection/classifier_codebase/logs/checkpoints/exp8/best.pth \
    --resume

"""
#############################


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import os
from utils import (
    load_split_train_test,
    plot_classes_preds,
    save_checkpoint,
    plot_confusion_matrix,
    load_ckp,
)
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import time
from model import Model
import sys
import configs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="this script trains the classification model"
    )

    parser.add_argument("--data_dir", required=True, help="training data path")
    parser.add_argument(
        "--log_dir", required=False, default="./logs", type=str, help="dir to save logs"
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="number of epochs to train a model"
    )
    parser.add_argument(
        "--save_interval", default=100, type=int, help="steps interval to save model"
    )
    parser.add_argument(
        "--print_interval", default=10, type=int, help="steps interval to print log"
    )
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument(
        "--test_split", default=0.2, type=float, help="test split out of 1.0"
    )
    parser.add_argument("--name", default="exp0", type=str, help="experiment name")
    parser.add_argument(
        "--checkpoint_path",
        default="./logs/best.pth",
        type=str,
        help="checkpoint path to resume training",
    )

    parser.add_argument(
        "--resume", action="store_true", help="resume from a checkpoint"
    )

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # tensorboard writter
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(os.path.join(args.log_dir, args.name))

    ##load data
    data_dir = args.data_dir
    trainloader, testloader = load_split_train_test(data_dir, args.batch_size)
    print(trainloader.dataset.classes)

    # sys.exit()

    ##load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_layers = len(configs.CLASSES)
    model_obj = Model(output_layers, device, args.lr)
    model, optimizer, criterion, scheduler = (
        model_obj.model,
        model_obj.optimizer,
        model_obj.criterion,
        model_obj.scheduler,
    )

    ## training loop
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = args.print_interval
    train_losses, test_losses = [], []
    best_accuracy = 0
    start_epoch = 0
    if args.resume:
        model, optimizer, start_epoch = load_ckp(args.checkpoint_path, model, optimizer)
        steps = len(trainloader) * start_epoch
        print(f"Model resumed from epoch:{start_epoch} steps: {steps}")

    try:
        print("Training Started")
        for epoch in range(start_epoch, epochs):
            for inputs, labels in trainloader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()

                    # for confusion matrix
                    predlist = torch.zeros(0, dtype=torch.long, device="cpu")
                    lbllist = torch.zeros(0, dtype=torch.long, device="cpu")

                    with torch.no_grad():
                        for inputs, labels in testloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()

                            ps = torch.exp(logps)

                            #                             _, preds = torch.max(ps, 1)

                            top_p, top_class = ps.topk(1, dim=1)
                            # Append batch prediction results
                            predlist = torch.cat([predlist, top_class.view(-1).cpu()])
                            lbllist = torch.cat(
                                [lbllist, labels.view(*top_class.shape).cpu()]
                            )

                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(
                                equals.type(torch.FloatTensor)
                            ).item()

                    # Confusion matrix
                    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
                    conf_fig = plot_confusion_matrix(
                        conf_mat, trainloader.dataset.classes
                    )
                    writer.add_figure(
                        "confusion matrix",
                        conf_fig,
                        global_step=epoch * len(trainloader) + steps,
                    )

                    # Per-class accuracy
                    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)

                    train_losses.append(running_loss / len(trainloader))
                    test_losses.append(test_loss / len(testloader))

                    # ...log the running loss
                    writer.add_scalar(
                        "loss/training_loss",
                        running_loss / print_every,
                        global_step=epoch * len(trainloader) + steps,
                    )
                    # ...log the test loss
                    writer.add_scalar(
                        "loss/test_loss",
                        test_loss / len(testloader),
                        global_step=epoch * len(trainloader) + steps,
                    )

                    # ...log the test Accuracy
                    writer.add_scalar(
                        "test Accuracy",
                        accuracy / len(testloader),
                        global_step=epoch * len(trainloader) + steps,
                    )

                    # ...log the LR
                    writer.add_scalar(
                        "Lr",
                        optimizer.param_groups[0]["lr"],
                        global_step=epoch * len(trainloader) + steps,
                    )

                    # ...log a Matplotlib Figure showing the model's predictions on a
                    # random mini-batch
                    writer.add_figure(
                        "predictions vs. actuals",
                        plot_classes_preds(model, inputs, labels),
                        global_step=epoch * len(trainloader) + steps,
                    )

                    print(
                        f"Epoch {epoch+1}/{epochs}.. "
                        f"Step :{steps}.. "
                        f"Lr: {optimizer.param_groups[0]['lr']:.6f}.."
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Test loss: {test_loss/len(testloader):.3f}.. "
                        f"Test accuracy: {accuracy/len(testloader):.3f}"
                    )
                    print(
                        f"Per class accuracy : {trainloader.dataset.classes} -> {class_accuracy}\n"
                    )

                    # save best model
                    if best_accuracy < (accuracy / len(testloader)):
                        best_accuracy = accuracy / len(testloader)
                        path = os.path.join(
                            args.log_dir,
                            "checkpoints",
                            args.name,
                        )
                        save_checkpoint(
                            path, "best", epoch, model, optimizer, train_losses
                        )
                        print(
                            f"Best checkpoint saved at :{path}, having accuracy :{best_accuracy} "
                        )

                    #                     scheduler.step(test_losses[-1])
                    running_loss = 0
                    model.train()

                if steps % args.save_interval == 0:
                    path = os.path.join(
                        args.log_dir,
                        "checkpoints",
                        args.name,
                    )
                    save_checkpoint(
                        path,
                        f"epochs-{epochs}-steps-{steps}",
                        epoch,
                        model,
                        optimizer,
                        train_losses,
                    )
                    print(f"checkpoint saved at :{path}")

            scheduler.step()

        path = os.path.join(args.log_dir, "checkpoints", args.name)
        save_checkpoint(path, "last", epoch, model, optimizer, train_losses)
        print(f"checkpoint saved at :{path}")

    except KeyboardInterrupt:
        path = os.path.join(args.log_dir, "checkpoints", args.name)
        save_checkpoint(path, "last", epoch, model, optimizer, train_losses)
        print(f"Training interrupted checkpoint saved at :{path}")