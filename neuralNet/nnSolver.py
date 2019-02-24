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
    
    def firstMove(self):
        # TODO: smartify and cite
        self.board.explore(0,0)
        return self.board
    
    def getAllSurroundingTiles(self, tile):
        # TODO: might need just a shared functions file for this nonsense
        surrounding = []

        row = tile.row
        col = tile.col

        surrounding.append(self.board.left(row, col))
        surrounding.append(self.board.right(row, col))
        surrounding.append(self.board.up(row, col))
        surrounding.append(self.board.down(row, col))
        surrounding.append(self.board.upLeft(row, col))
        surrounding.append(self.board.upRight(row, col))
        surrounding.append(self.board.downLeft(row, col))
        surrounding.append(self.board.downRight(row, col))
        
        return surrounding
    
    def mark(self, row, col):
        """ Marks the tile at the row and column as a bomb
        and updates the remaining value of surrounding tiles """
        self.board.mark(row, col)
        self.unmarkedBombs -= 1
        tilesToReduce = self.getAllSurroundingTiles(Tile(0, row, col))
        for t in tilesToReduce:
            if t is not None and t.remainingValue != 0 and t.value != BOMB:
                self.board.board[t.row][t.col].remainingValue -= 1
    
    def move(self):
        # TODO: prioritize exploring so that we don't overmark as much
            # If there is a move that is over the threshold pick the max move
            # If there isn't then mark something that is over the threshold
            # Else explore far away
        # TODO: Sometimes explore other places
        validTiles = transformBoard(self.board)
        nnInput = [x["nn"] for x in validTiles]
        tiles = [x["tile"] for x in validTiles]
        probs = self.determineProbs(torch.Tensor(nnInput))
        confidence = [abs(x[0] - x[1]) for x in probs]
        index = np.argmax(confidence)
        tile = tiles[index]
        move = 0 if probs[index][0] > probs[index][1] else 1 # 0 - explore, 1 - bomb

        if abs(probs[index][0] - probs[index][1]) < CERTAINTY_THRESHOLD:
            print("-"*16, "I AM UNCERTAIN ABOUT THIS MOVE!", "-"*16)
        print("Chosen tile:", tile.row, ",", tile.col)
        print("Chosen move:", "Mark" if move == 1 else "Explore")
        if move == 1:
            # FIXME:? make sure it's not yet marked
            self.mark(tile.row, tile.col)
        else:
            self.board.explore(tile.row, tile.col)
        
        return self.board
        

    def determineProbs(self, tiles):
        tiles = tiles.view((-1, 12, 5, 5)).cuda()
        probs = self.net.classifyTiles(tiles)
        return probs
        
if __name__ == "__main__":
    nns = NeuralNetSolver()
    print(nns.firstMove())
    while nns.unmarkedBombs > 0:
        print(nns.move())


    # TESTING
    # data, labels = generateTrainingData()
    # nns.net.eval()
    # with torch.no_grad():
    #     output = nns.net.forward(data)
    #     mistakes = (labels != torch.argmax(output, dim=1))
    #     output = output.cpu().numpy()
    #     mistakes = mistakes.cpu().numpy().astype(bool)
    #     labels = labels.cpu().numpy()
    #     mistakePredValues = output[mistakes]
    #     correctPredValues = output[~mistakes]
    #     correctDiff = [abs(x[0] - x[1]) for x in correctPredValues]
    #     mistakeDiff = [abs(x[0] - x[1]) for x in mistakePredValues]

    #     print("Max correct diff:", max(correctDiff))
    #     print("Max mistake diff:", max(mistakeDiff))
    #     print()
    #     print("Min correct diff:", min(correctDiff))
    #     print("Min mistake diff:", min(mistakeDiff))
    #     print()
    #     print("Avg correct diff:", sum(correctDiff) / len(correctDiff))
    #     print("Avg mistake diff:", sum(mistakeDiff) / len(mistakeDiff))
    #     print()

    #     #Drop values where the difference is under threshold
    #     threshold = lambda x: abs(x[0] - x[1]) > CERTAINTY_THRESHOLD
    #     mask = [threshold(x) for x in output]  # TODO: loops bad
    #     certainOut = output[mask]
    #     certainLabels = labels[mask]
    #     certainAcc = np.mean(certainLabels == np.argmax(certainOut, axis=1))
    #     print("Thresholded accuracy:", certainAcc)
    #     print("Number certain results:", certainLabels.shape)
    #     print("Percentage certain results:", certainLabels.shape[0] / BATCH_SIZE)

    # nns.net.train()