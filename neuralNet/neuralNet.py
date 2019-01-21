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
from game.constants import BOMB, HEIGHT, MAX_BOMBS, TRAINING_DATA_FILE, WIDTH
from game.tile import Tile


class NeuralNetSolver:
    def __init__(self):
        # TESTING
        # self.board = Board()
        # self.board.board = [[Tile(0, r, c) for c in range(WIDTH)] for r in range(HEIGHT)]
        # self.board.board[0][1] = Tile(BOMB, 0, 1)
        # self.board.board[1][1] = Tile(BOMB, 1, 1)
        # for r in range(WIDTH):
        #     for c in range(HEIGHT):
        #         self.board.updateCounts(r, c)
        
        self.board = Board()
        self.unmarkedBombs = MAX_BOMBS
        self.X, self.y = self.loadTrainingData() # X is data, y is labels
        
        # TODO: validation + test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.print5x5(self.X_train[0], self.y_train[0])
    
    def print5x5(self, x, y):
        """ Prints an arrangement around a tile TODO: use .format()"""
        print("Y", y)
        isBomb = " * " if y == 1 else " _ "

        print()

        print("{:1}".format(x[7]*8), "{:1}".format(x[8]*8), "{:1}".format(x[9]*8), "{:1}".format(x[11]*8), "{:1}".format(x[12]*8))
        print("{:1}".format(x[6]*8), "{:1}".format(x[4]*8), "{:1}".format(x[2]*8), "{:1}".format(x[10]*8), "{:1}".format(x[13]*8))
        print("{:1}".format(x[5]*8), "{:1}".format(x[0]*8), isBomb, "{:1}".format(x[1]*8), "{:1}".format(x[23]*8))
        print("{:1}".format(x[15]*8), "{:1}".format(x[14]*8), "{:1}".format(x[3]*8), "{:1}".format(x[18]*8), "{:1}".format(x[22]*8))
        print("{:1}".format(x[16]*8), "{:1}".format(x[17]*8), "{:1}".format(x[19]*8), "{:1}".format(x[20]*8), "{:1}".format(x[21]*8))

    
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
        X = [[0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5],]
        p = self.clf.predict(X)
        print("CUSTOM prediction:", p)
        print("Layout")
        self.print5x5(X[0], p)
    
if __name__ == "__main__":
    nns = NeuralNetSolver()
    nns.train()
    # nns.test()
    nns.customTest()
