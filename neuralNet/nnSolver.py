import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from game.board import Board
from game.tile import Tile
from game.constants import *
from itertools import permutations, chain
from sympy.utilities.iterables import multiset_permutations
from math import factorial
from twoD_nn import miniNet
from generateTrainingData import transformBoard, generateTrainingData
import numpy as np
import torch

#TESTING
from pprint import pprint

class NeuralNetSolver:
    def __init__(self):
        self.board = Board()
        self.unmarkedBombs = MAX_BOMBS
        self.net = miniNet().cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR, weight_decay=REG)
        self.net.load(self.optimizer) # TODO: ensure this works

        # Set up remainingValue to be reduced whenever a bomb is marked
        for i, r in enumerate(self.board.board):
            for j, _ in enumerate(r):
                self.board.board[i][j].remainingValue = self.board.board[i][j].value
    
    def firstMove(self):
        # TODO: smartify and cite
        self.board.explore(0,0)
    
    def mark(self, row, col):
        """ Marks the tile at the row and column as a bomb
        and updates the remaining value of surrounding tiles """
        self.board.mark(row, col)
        self.unmarkedBombs -= 1
        tilesToReduce = self.getAllSurroundingTiles(Tile(0, row, col))
        for t in tilesToReduce:
            if t is not None and t.value != 0 and t.value != BOMB:
                self.board.board[t.row][t.col].remainingValue -= 1
    
    def move(self):
        validTiles = transformBoard(self.board)
        nnInput = [x["nn"] for x in validTiles]
        tiles = [x["tile"] for x in validTiles]
        probs = self.determineProbs(nnInput)
        # NEXTTIME continue here

    def determineProbs(self, tiles):
        probs = np.zeros((len(tiles), 2))
        for i, _ in enumerate(probs):
            probs[i] = self.net.classifyTile(tiles[i])
        return probs
        
if __name__ == "__main__":
    nns = NeuralNetSolver()
    data, labels = generateTrainingData()
    nns.net.eval()
    with torch.no_grad():
        output = nns.net.forward(data)
        mistakes = (labels != torch.argmax(output, dim=1))
        output = output.cpu().numpy()
        mistakes = mistakes.cpu().numpy().astype(bool)
        labels = labels.cpu().numpy()
        mistakePredValues = output[mistakes]
        correctPredValues = output[~mistakes]
        correctDiff = [abs(x[0] - x[1]) for x in correctPredValues]
        mistakeDiff = [abs(x[0] - x[1]) for x in mistakePredValues]

        print("Max correct diff:", max(correctDiff))
        print("Max mistake diff:", max(mistakeDiff))
        print()
        print("Min correct diff:", min(correctDiff))
        print("Min mistake diff:", min(mistakeDiff))
        print()
        print("Avg correct diff:", sum(correctDiff) / len(correctDiff))
        print("Avg mistake diff:", sum(mistakeDiff) / len(mistakeDiff))
        print()

        #Drop values where the difference is under threshold
        threshold = lambda x: abs(x[0] - x[1]) > CERTAINTY_THRESHOLD
        # threshold = np.vectorize(threshold)
        mask = [threshold(x) for x in output]
        certainOut = output[mask] # TODO: loops bad
        certainLabels = labels[mask] # TODO: loops bad
        certainAcc = np.mean(certainLabels == np.argmax(certainOut, axis=1))
        print("Thresholded accuracy:", certainAcc)
        print("Number certain results:", certainLabels.shape)
        print("Percentage certain results:", certainLabels.shape[0] / BATCH_SIZE)

    nns.net.train()