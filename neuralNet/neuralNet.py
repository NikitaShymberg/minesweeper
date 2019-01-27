# TODO: run on GPU :O

import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

import h5py
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game.board import Board
from game.constants import BOMB, MAX_BOMBS, TRAINING_DATA_FILE
from game.tile import Tile
from generateTrainingData import generateTrainingData
import tensorflow as tf


class miniNet(nn.Module):
    def __init__(self):
        super(miniNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 12, 3)
        self.conv2 = nn.Conv2d(12, 12, 3)
        self.lin3 = nn.Linear(12, 2) # Output 2 to see both bomb and not bomb
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.lin3(x))
        return x
    
    def test(self):
        y_pred = self.clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", acc)
    
if __name__ == "__main__":
    mini = miniNet()
    print(mini)
    # print(mini.parameters())
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mini.parameters(), lr=0.01)

    running_loss = 0.0
    data, labels = generateTrainingData()
    data = tf.convert_to_tensor(data) # BUG
    for i, datum in enumerate(data):
        label = labels[i]

        optimizer.zero_grad() # ???
        outputs = mini(datum)
        l = loss(outputs, label)
        l.backward()
        optimizer.step()

        print(l.item())


