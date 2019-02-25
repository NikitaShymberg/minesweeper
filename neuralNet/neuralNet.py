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
        self.conv1 = nn.Conv2d(12, 500, 3)
        self.lin2 = nn.Linear(500 * 9, 500*9)
        self.lin3 = nn.Linear(500 * 9, 500)
        self.lin4 = nn.Linear(500, 250)
        self.lin5 = nn.Linear(250, 100)
        self.lin6 = nn.Linear(100, 2)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 500 * 9) # Reshape to fit into Linear layer
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = torch.relu(self.lin4(x))
        x = torch.relu(self.lin5(x))
        x = torch.relu(self.lin6(x))
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
            return self.forward(data) # FIXME: does this work with one thing at a time?
    
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
    reg = 10**np.random.uniform(-10, -5, 15)
    lr = 10**np.random.uniform(-6, -4, 15)

    for trial in range(15):
        REG = reg[trial]
        LR = lr[trial]

        if torch.cuda.is_available():
            mini = miniNet().cuda()
        else:
            mini = miniNet()
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

            if epoch % (EPOCHS // 10) == 0 and epoch != 0:
                # print("EPOCH: ", epoch, "Training Loss:", loss.item())
                # print("EPOCH: ", epoch, "Training Accuracy:", acc)
                tr_writer.add_scalar("data/loss_lr={}_reg={}:".format(LR, REG), loss, epoch)
                tr_writer.add_scalar("data/accuracy_lr={}_reg={}:".format(LR, REG), acc, epoch)

                # val_data, val_labels = generateTrainingData()
                # val_loss, val_acc = mini.test(val_data, val_labels, criterion)
                # print("EPOCH: ", epoch, "Validation Loss:", val_loss.item())
                # print("EPOCH: ", epoch, "Validation Accuracy:", val_acc)
                # tr_writer.add_scalar("data/val_loss", val_loss, epoch)
                # tr_writer.add_scalar("data/val_accuracy", val_acc, epoch)

                mini.save(epoch, optimizer)
        
        val_data, val_labels = generateTrainingData()
        val_loss, val_acc = mini.test(val_data, val_labels, criterion)
        print("FINAL Validation Loss_lr={}_reg={}:".format(LR, REG), val_loss.item())
        print("FINAL Validation Accuracy_lr={}_reg={}:".format(LR, REG), val_acc)


