"""
NOTE: this is a backup of the old conv1d neural net. It's likely to not be as good
as the upcoming replacement, but I'll keep it around in case it is.
I saved the old model trained 300,000 iterations on batches of 2,000 setups
"""
import sys, os
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

import h5py
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.board import Board
from game.constants import *
from game.tile import Tile
from generateTrainingData import generateTrainingData
from tensorboardX import SummaryWriter

class miniNet(nn.Module):
    def __init__(self):
        super(miniNet, self).__init__()
        self.conv1 = nn.Conv1d(12, 500, 3)
        self.conv2 = nn.Conv1d(500, 500, 3)
        self.lin3 = nn.Linear(500 * 20, 2)
    
    def forward(self, x):
        # print("x shape1", x.shape)
        x = torch.relu(self.conv1(x))
        # print("x shape2", x.shape)
        x = torch.relu(self.conv2(x))
        # print("x shape3", x.shape)
        x = x.view(-1, 500 * 20) # Reshape to fit into Linear layer
        # print("x shape4", x.shape)
        x = torch.relu(self.lin3(x))
        # print("x shape5", x.shape)
        return x
    
    def train_model(self, data, labels, criterion, optimizer):
        optimizer.zero_grad()

        output = self.forward(data)
        # print("Predicted:", torch.argmax(output, dim=1))
        # print("Actual:   ", labels)
        loss = criterion(output, labels)
        accs = torch.mean((labels == torch.argmax(output, dim=1)).float())
        loss.backward()
        optimizer.step()

        return loss, accs
    
    def test(self, data, labels, criterion):
        self.eval()
        with torch.no_grad():
            output = self.forward(data)
            loss = criterion(output, labels)
            accs = torch.mean((labels == torch.argmax(output, dim=1)).float())
        
        self.train()
        return loss, accs
    
    def classifyTile(self, data):
        self.eval()
        with torch.no_grad():
            return self.forward(data) # TODO: does this work with one thing at a time?
    
    def save(self, epoch, optimizer):
        torch.save({
            "state": self.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict()
        }, CHECKPOINT_FILE)
    
    def load(self, optimizer):
        checkpoint = torch.load(CHECKPOINT_FILE)
        self.load_state_dict(checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint["epoch"]

    
if __name__ == "__main__":
    mini = miniNet().cuda()
    # print(mini)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mini.parameters(), lr=LR, weight_decay=REG)
    # if os.path.isfile(CHECKPOINT_FILE):
        # first_epoch = mini.load(optimizer)
    # else:
    first_epoch = 0

    tr_writer = SummaryWriter("runs/training")
    val_writer = SummaryWriter("runs/validation")
    for epoch in range(first_epoch, EPOCHS):
        data, labels = generateTrainingData()
        loss, acc = mini.train_model(data, labels, criterion, optimizer)

        if epoch % (EPOCHS // 1000) == 0 and epoch != 0:
            print("EPOCH: ", epoch, "Training Loss:", loss.item())
            print("EPOCH: ", epoch, "Training Accuracy:", acc)
            tr_writer.add_scalar("data/loss", loss, epoch)
            tr_writer.add_scalar("data/accuracy", acc, epoch)

            val_data, val_labels = generateTrainingData()
            val_loss, val_acc = mini.test(val_data, val_labels, criterion)
            print("EPOCH: ", epoch, "Validation Loss:", val_loss.item())
            print("EPOCH: ", epoch, "Validation Accuracy:", val_acc)
            tr_writer.add_scalar("data/val_loss", val_loss, epoch)
            tr_writer.add_scalar("data/val_accuracy", val_acc, epoch)

            mini.save(epoch, optimizer)
    
    val_data, val_labels = generateTrainingData()
    val_loss, val_acc = mini.test(val_data, val_labels, criterion)
    print("FINAL Validation Loss:", val_loss.item())
    print("FINAL Validation Accuracy:", val_acc)


