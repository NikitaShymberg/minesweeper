import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from game.board import Board
from game.tile import Tile
from game.constants import MAX_BOMBS, BOMB, WIDTH, HEIGHT
from itertools import permutations, chain
from sympy.utilities.iterables import multiset_permutations
from math import factorial
from neuralNet import miniNet
from generateTrainingData import transformBoard
import numpy as np

#TESTING
from pprint import pprint

class BruteForceSolver:
    def __init__(self):
        self.board = Board()
        self.unmarkedBombs = MAX_BOMBS
        self.net = miniNet()
        self.optimizer = optim.Adam(mini.parameters(), lr=LR, weight_decay=REG)
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
        probs = self.determineProbs(nnInput)
        # NEXTTIME continue here

    def determineProbs(self, tiles):
        probs = np.zeros((len(tiles), 2))
        for i, _ in enumerate(probs):
            probs[i] = self.net.classifyTile(tiles[i])
        return probs
        
