import os
import sys
import time

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Visualization

import seaborn as sns
import matplotlib.pyplot as plt

# PyTorch
import torch as T
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms
current_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(current_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
#path = 'data/dogs/'
#path = 'data/poke/'
#path = 'data/Brain/'
#path = 'CIFAR10/ANN_baseline/Dataset'
path = 'data/mnist/'
# image_s = 224
# batch_size=64
batch_size = 64
image_s = 114


class Model(nn.Module):

    def __init__(self, inception_model, resnet50_model):
        super(Model, self).__init__()

        self.inception_model = inception_model
        self.resnet50_model = resnet50_model

        self.output = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(4096, 120)
        )

        self.to(device)
        # Optimizer
        self.optim = T.optim.SGD(self.output.parameters(), lr=0.005, momentum=0.9)
        # Loss
        self.criterion = T.nn.CrossEntropyLoss()
        # Scheduler
        self.scheduler = T.optim.lr_scheduler.StepLR(self.optim, step_size=7, gamma=0.1)

    def forward(self, x):
        X1 = self.inception_model(x)
        X2 = self.resnet50_model(x)

        X1 = X1.view(X1.size(0), -1)
        X2 = X2.view(X2.size(0), -1)

        X = T.cat([X1, X2], dim=1)

        P = self.output(X)

        return P

    def get_weights(self):
        return self.output.state_dict()

    def load_weights(self, weights):
        self.output.load_state_dict(weights)


def train_model(train_dl, val_dl, model, epochs=20):
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    # Best validation accuracy
    best_val_loss = 1_000_000.0
    # Get initial weights
    weights = model.get_weights()
    for epoch in range(epochs):
        print("=" * 20, "Epoch: ", str(epoch + 1), "=" * 20)
        since = time.time()
        train_correct_pred = 0
        val_correct_pred = 0
        train_acc = 0
        val_acc = 0
        train_loss = 0
        val_loss = 0

        # Set to training mode
        model.train()

        for x, y in train_dl:
            # Convert data to Tensor
            x = x.clone().detach().to(device).requires_grad_(True)
            y = y.clone().detach().long().to(device)
            # Reset gradients
            model.optim.zero_grad()
            # Predict
            preds = model(x)

            # Compute the loss
            loss = model.criterion(preds, y)

            # Compute the gradients
            loss.backward()
            # Update weights
            model.optim.step()
            # Count the correct predictions
            preds = T.argmax(preds, dim=1)
            train_correct_pred += (preds.long().unsqueeze(1) == y.unsqueeze(1)).sum().item()

            train_loss += loss.item()

        train_acc = train_correct_pred / len(train_dl.dataset)

        train_acc_history.append(train_acc)

        train_loss_history.append(train_loss)

        # Switch to evaluation mode
        model.eval()

        with T.no_grad():
            for x, y in val_dl:
                # Convert data to Tensor
                x = x.clone().detach().to(device)
                y = y.clone().detach().long().to(device)
                # Predict
                preds = model(x)
                # Compute the loss
                loss = model.criterion(preds, y)

                val_loss += loss.item()
                # Count the correct predictions
                preds = T.argmax(preds, dim=1)

                val_correct_pred += (preds.long().unsqueeze(1) == y.unsqueeze(1)).sum().item()

        model.scheduler.step()

        val_acc = val_correct_pred / len(val_dl.dataset)

        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        # Save the weights of the best model
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            weights = model.get_weights()
        time_elapsed = time.time() - since
        print("Train acc: {:.4f} | Train Loss: {:.4f} | Validation acc: {:.4f} | Validation Loss: {:.4f}".format(
            train_acc, train_loss, val_acc, val_loss))
        print(f"Time: {time_elapsed}")
    # Load best model
    model.load_weights(weights)

    return [train_acc_history, train_loss_history, val_acc_history, val_loss_history], model


if __name__ == '__main__':
    if "poke" in path.lower():
        from data.data_loader_custom import build_data

        name1 = "poke"
        batch_size = 64
        image_s = 75
        l_features = int((image_s ** 2) / 2)

    elif "brain" in path.lower():
        from data.data_loader_custom import build_data

        name1 = "brain"
        batch_size = 32
        image_s = 75
        l_features = int((image_s ** 2) / 2)

    elif "dogs" in path.lower():
        from data.data_loader_custom import build_data

        name1 = "dogs"
        batch_size = 64
        image_s = 75
        l_features = int(((round(image_s / 32) * 32) ** 2) / 2)
    elif "mnist" in path.lower():
        from data.data_loader_MNISTEdit import build_data

        name1 = "mnist"
        batch_size = 64
        image_s = 75
        l_features = int(((round(image_s / 32) * 32) ** 2) / 2)
    else:
        from data.data_loader_cifar10Edit import build_data

        name1 = "Cifar10"
        batch_size = 64
        image_s = 75
        # E:\Projects\Python\RP2\Hybrid-Coding-SNN-main\CIFAR10\ANN_baseline\Dataset
        l_features = int((image_s ** 2) / 2)
    inception = models.inception_v3(pretrained=True)

    inception_model = nn.Sequential(
        inception.Conv2d_1a_3x3,
        inception.Conv2d_2a_3x3,
        inception.Conv2d_2b_3x3,
        inception.maxpool1,
        inception.Conv2d_3b_1x1,
        inception.Conv2d_4a_3x3,
        inception.maxpool2,
        inception.Mixed_5b,
        inception.Mixed_5c,
        inception.Mixed_5d,
        inception.Mixed_6a,
        inception.Mixed_6b,
        inception.Mixed_6c,
        inception.Mixed_6d,
        inception.Mixed_6e,
        inception.Mixed_7a,
        inception.Mixed_7b,
        inception.Mixed_7c,
        inception.avgpool
    )

    resnet50 = models.resnet50(pretrained=True)

    resnet50_model = nn.Sequential(
        resnet50.conv1,
        resnet50.bn1,
        resnet50.relu,
        resnet50.maxpool,
        resnet50.layer1,
        resnet50.layer2,
        resnet50.layer3,
        resnet50.layer4,
        resnet50.avgpool
    )

    # Freeze parameters of pretrained models
    for param in resnet50_model.parameters():
        param.requires_grad = False

    for param in inception_model.parameters():
        param.requires_grad = False

    train_dl, validation_dl, classes = build_data(dpath=path, batch_size=batch_size, image_s=image_s, workers=4)
    print(train_dl.dataset)
    model = Model(inception_model, resnet50_model)
    epochs=8
    history, model = train_model(train_dl, validation_dl, model, epochs=epochs)

    # Training and Validation Results
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].plot(range(epochs), history[0], label="Training")
    axs[0].plot(range(epochs), history[2], label="Validation")
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].grid(True)

    axs[1].plot(range(epochs), history[1], label="Training")
    axs[1].plot(range(epochs), history[3], label="Validation")
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True)

    plt.suptitle("Training and Validation Results of Model")
    plt.legend()
    plt.savefig(f'Training and Validation Results of Model 3-{name1}.png')
