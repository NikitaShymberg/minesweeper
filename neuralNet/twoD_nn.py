import sys, os
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer


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
        inputSize = 11 if MODEL == "2dnnNEW" else 12
        self.conv1 = nn.Conv2d(inputSize, 500, 3)
        self.conv2 = nn.Conv2d(500, 500, 3)
        self.lin3 = nn.Linear(500 * 1, 2)
    
    def forward(self, x):
        # print("x shape1", x.shape)
        x = torch.relu(self.conv1(x))
        # print("x shape2", x.shape)
        x = torch.relu(self.conv2(x))
        # print("x shape3", x.shape)
        x = x.view(-1, 500 * 1) # Reshape to fit into Linear layer
        # print("x shape4", x.shape)
        x = torch.relu(self.lin3(x)) # FIXME This should be a sigmoid(?) to make it a probability!
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
    
    def classifyTiles(self, data):
        self.eval()
        # data.view(1, )
        with torch.no_grad():
            result = self.forward(data) # FIXME: does this work with one thing at a time?
        self.train()
        return result
    
    def save(self, epoch, optimizer):
        torch.save({
            "state": self.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict()
        }, CHECKPOINT_FILE)
    
    def load(self, optimizer):
        if torch.cuda.is_available():
            checkpoint = torch.load(CHECKPOINT_FILE)
        else:
            checkpoint = torch.load(CHECKPOINT_FILE, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint["epoch"]

    
if __name__ == "__main__":
    
    # TESTING
    # net = miniNet().cuda()
    # optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=REG)
    # net.load(optimizer)
    # criterion = nn.CrossEntropyLoss()
    # data, labels = generateTrainingData()
    # net.eval()
    # with torch.no_grad():
    #     output = net.forward(data)
    #     mistakes = (labels != torch.argmax(output, dim=1))
    #     output = output.cpu().numpy()
    #     mistakes = mistakes.cpu().numpy().astype(bool)
    #     labels = labels.cpu().numpy()
    #     mistakePredValues = output[mistakes]
    #     correctPredValues = output[~mistakes]
    #     data = data.cpu().numpy()
    #     for q in range(10):
    #         print("Predicted value:", mistakePredValues[q])
    #         print("Data:")
    #         values = np.zeros((5, 5))
    #         for i, row in enumerate(data[q].reshape(5, 5, 11)):
    #             for j, val in enumerate(row):
    #                 values[i][j] = np.where(val == 1)[0]
    #                 if values[i][j] > 1:
    #                     values[i][j] -= 1
    #                 elif values[i][j] == 1:
    #                     values[i][j] = 9
    #         print(values)
    # net.train()
    # TESTING

    if torch.cuda.is_available():
        mini = miniNet().cuda()
    else:
        mini = miniNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mini.parameters(), lr=LR, weight_decay=REG)
    if os.path.isfile(CHECKPOINT_FILE):
        print("Loading previous file...")
        first_epoch = mini.load(optimizer)
    else:
        first_epoch = 0

    tr_writer = SummaryWriter("runs/training")
    val_writer = SummaryWriter("runs/validation")
    for epoch in range(first_epoch, EPOCHS):
        data, labels = generateTrainingData()
        loss, acc = mini.train_model(data, labels, criterion, optimizer)

        if epoch % (EPOCHS // 100000) == 0 and epoch != 0:
            print("EPOCH: ", epoch, "Training Loss:  ", loss.item())
            print("EPOCH: ", epoch, "Training Accuracy:  ", acc.item())
            tr_writer.add_scalar("data/loss", loss.item(), epoch)
            tr_writer.add_scalar("data/accuracy", acc.item(), epoch)

            val_data, val_labels = generateTrainingData()
            val_loss, val_acc = mini.test(val_data, val_labels, criterion)
            print("EPOCH: ", epoch, "Validation Loss:", val_loss.item())
            print("EPOCH: ", epoch, "Validation Accuracy:", val_acc.item())
            tr_writer.add_scalar("data/val_loss", val_loss.item(), epoch)
            tr_writer.add_scalar("data/val_accuracy", val_acc.item(), epoch)

            mini.save(epoch, optimizer)
    
    val_data, val_labels = generateTrainingData()
    val_loss, val_acc = mini.test(val_data, val_labels, criterion)
    print("FINAL Validation Loss", val_loss.item())
    print("FINAL Validation Accuracy", val_acc.item())