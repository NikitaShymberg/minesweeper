# NOTE: I wonder how bomb density will affect this stuff

import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

import h5py
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from game.board import Board
from game.constants import BOMB, MAX_BOMBS, TRAINING_DATA_FILE
from game.tile import Tile


class NeuralNetSolver:
    def __init__(self):
        self.board = Board()
        self.unmarkedBombs = MAX_BOMBS
        self.X, self.y = self.loadTrainingData() # X is data, y is labels
        
        # TODO: validation + test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        print(self.X_train)
        self.print5x5(self.X_train[0], self.y_train[0]) # TESTING

    def print5x5(self, x, y):
        """ Prints a 5x5 instance of data TODO: work with onehot """
        isBomb = "  * " if y == 1 else "  _ "

        # print("{:4}".format(x[7]*8), "{:4}".format(x[8]*8), "{:4}".format(x[9]*8), "{:4}".format(x[11]*8), "{:4}".format(x[12]*8))
        # print("{:4}".format(x[6]*8), "{:4}".format(x[4]*8), "{:4}".format(x[2]*8), "{:4}".format(x[10]*8), "{:4}".format(x[13]*8))
        # print("{:4}".format(x[5]*8), "{:4}".format(x[0]*8), isBomb, "{:4}".format(x[1]*8), "{:4}".format(x[23]*8))
        # print("{:4}".format(x[15]*8), "{:4}".format(x[14]*8), "{:4}".format(x[3]*8), "{:4}".format(x[18]*8), "{:4}".format(x[22]*8))
        # print("{:4}".format(x[16]*8), "{:4}".format(x[17]*8), "{:4}".format(x[19]*8), "{:4}".format(x[20]*8), "{:4}".format(x[21]*8))
        print()

    def loadTrainingData(self):
        with h5py.File(TRAINING_DATA_FILE, "r") as f:
            return f["data"][:], f["class"][:]
    
    def train(self):
        self.clf = MLPClassifier(hidden_layer_sizes=(24, 24), max_iter=1)
        self.clf.fit(self.X_train, self.y_train)
    
    def test(self):
        y_pred = self.clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", acc)

    def customTest(self):
        X = [[0.125, 0.125, 0.125, 0.125, 0.125, 0, 0, 0, 0, 0, 0.125, 0, 0, 0, 0.125, 0, 0, 0, 0.125, 0, 0, 0, 0, 0],]
        # X = [[0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5],]
        p = self.clf.predict(X)[0]
        print("My prediction:", p)
        print("Tile layout:")
        self.print5x5(X[0], p)
    
if __name__ == "__main__":
    nns = NeuralNetSolver()
    nns.train()
    # nns.test()
    nns.customTest()
