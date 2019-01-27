# TODO: run on GPU :O

import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

import h5py
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game.board import Board
from game.constants import BOMB, MAX_BOMBS, TRAINING_DATA_FILE, BATCH_SIZE
from game.tile import Tile
from generateTrainingData import generateTrainingData
import tensorflow as tf


class miniNet(nn.Module):
    def __init__(self):
        super(miniNet, self).__init__()
        self.conv1 = nn.Conv1d(11, 11, 3)
        self.conv2 = nn.Conv1d(11, 11, 3)
        self.lin3 = nn.Linear(11 * BATCH_SIZE, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 11 * BATCH_SIZE) # Reshape to fit into Linear layer
        x = F.relu(self.lin3(x))
        return x
    
    def train(self, data, labels, criterion, optimizer):
        optimizer.zero_grad()

        output = self(data)
        print("output shape:", output.shape)
        print("labels shape:", labels.shape)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        return loss
    
    def test(self):
        y_pred = self.clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", acc)
    
if __name__ == "__main__":
    mini = miniNet()
    print(mini)
    # print(mini.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mini.parameters(), lr=0.01) # TODO: lr

    for epoch in range(100):
        running_loss = 0.0
        data, labels = generateTrainingData()
        data = data.reshape(BATCH_SIZE, 11, 24)
        data = torch.from_numpy(data).float() #TODO: do this in generateTrainingData()
        labels = torch.from_numpy(labels)

        print("data shape:", data.shape)
        print("data type:", type(data))
        print("labels shape:", labels.shape)
        # data = tf.unstack(data)
        loss = mini.train(data, labels, criterion, optimizer)

        print("Loss:", loss.item())
    
    #TESTING
    testData, testLabels = generateTrainingData()
    testData = testData.reshape(BATCH_SIZE, 11, 24)
    testData = torch.from_numpy(testData).float()
    allegedLabels = mini(testData)
    _, allegedLabels = torch.max(allegedLabels, 1)
    print("ACTUAL:", testLabels)
    print("PREDICTED:", allegedLabels)


