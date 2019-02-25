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
        if torch.cuda.is_available():
            self.net = miniNet().cuda()
        else:
            self.net = miniNet()
            
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
            if t is not None and t.remainingValue > 0 and t.value != BOMB:
                self.board.board[t.row][t.col].remainingValue -= 1
    
    def move(self):
        # TODO: clean/reafactor
        # TODO: prioritize exploring so that we don't overmark as much
            # If there is a move that is over the threshold pick the max move
            # If there isn't then mark something that is over the threshold
            # Else explore far away
        # FIXME: remaining value is borked asf as well
        validTiles = transformBoard(self.board)
        nnInput = [x["nn"] for x in validTiles]
        tiles = [x["tile"] for x in validTiles]
        probs = self.determineProbs(torch.Tensor(nnInput)).cpu().numpy()
        confidence = [abs(x[0] - x[1]) for x in probs]
        for i in range(len(confidence)):
            validTiles[i]["confidence"] = confidence[i]

        isExplore = lambda x: x[0] > x[1]
        explores = np.array([isExplore(x) for x in probs])
        marks = ~explores
        explores = validTiles[explores]
        marks = validTiles[marks]
        
        exploreMove = None
        markMove = None
        dictToList = lambda d, key: [x[key] for x in d]
        if len(explores) > 0:
            exploreMove = explores[np.argmax(dictToList(explores, "confidence"))]

        if len(marks) > 0:
            # TESTING: I'm not sure maybe this is cheating, this is waht it was before:
            # markMove = marks[np.argmax(dictToList(marks, "confidence"))]
            # TODO: maybe cheat more and straight up explore the tile
            valid = False
            while not valid and len(marks) > 0:
                index = np.argmax(dictToList(marks, "confidence"))
                markMove = marks[index]
                surroundingTiles = self.getAllSurroundingTiles(markMove["tile"])
                surroundingValues = [t.remainingValue for t in surroundingTiles if t is not None and t.explored]
                if 0 in surroundingValues:
                    marks = np.delete(marks, index)
                    print("-"*16, "Tried to make a stupid move :(", "-"*16, ":", "MARK:", markMove["tile"].row, markMove["tile"].col)
                else:
                    valid = True
        
        if exploreMove is not None and exploreMove["confidence"] > MOVE_CERTAINTY_THRESHOLD:
            move = 0
            tile = exploreMove["tile"]
        elif markMove is not None and markMove["confidence"] > MARK_CERTAINTY_THRESHOLD:
            move = 1
            tile = markMove["tile"]
        else:
            # TODO: consider unexplored tiles adjacent to validTiles ?
            print("-"*16, "I AM UNCERTAIN ABOUT THIS MOVE!", "-"*16)
            farTiles = []
            for r in self.board.board:
                for tile in r:
                    if not tile.explored:
                        farTiles.append(tile)
            move = 0
            if len(farTiles) > 0:
                tile = np.random.choice(farTiles)
            else:
                tile = exploreMove["tile"]

        print("Chosen tile:", tile.row, tile.col)
        print("Chosen move:", "Mark" if move == 1 else "Explore")
        if move == 1:
            self.mark(tile.row, tile.col)
        else:
            self.board.explore(tile.row, tile.col)
        
        return self.board
        

    def determineProbs(self, tiles):
        """ Returns the output of the nn on the given tiles 
            As well as a mask for the explores and marks
        """
        if torch.cuda.is_available():
            tiles = tiles.view((-1, 12, 5, 5)).cuda()
        else:
            tiles = tiles.view((-1, 12, 5, 5))

        probs = self.net.classifyTiles(tiles)
        return probs
        
if __name__ == "__main__":
    nns = NeuralNetSolver()
    print(nns.firstMove())
    while nns.unmarkedBombs > 0: #TODO: have a real win condition check, same for bruteForce
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