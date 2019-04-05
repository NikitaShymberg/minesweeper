import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from time import perf_counter
from game.board import Board
from game.tile import Tile
from game.constants import *
from itertools import permutations, chain
from sympy.utilities.iterables import multiset_permutations
from math import factorial
from neuralNet.twoD_nn import miniNet
from neuralNet.generateTrainingData import transformBoard, generateTrainingData, printData
import numpy as np
import torch


class NeuralNetSolver:
    def __init__(self, rows, cols, bombs):
        self.board = Board(rows, cols, bombs)
        self.unmarkedBombs = bombs
        if torch.cuda.is_available():
            self.net = miniNet().cuda()
        else:
            self.net = miniNet()
            
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR, weight_decay=REG)
        self.net.load(self.optimizer)
    
    def firstMove(self):
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
        """
        Performs the optimal move, if the move was successful returns the
        current board, else returns None to signify that a mine was triggered
        """
        # TODO: clean/reafactor
        validTiles = transformBoard(self.board)
        nnInput = [x["nn"] for x in validTiles]
        tiles = [x["tile"] for x in validTiles]
        probs = self.determineProbs(torch.Tensor(nnInput)).cpu().numpy()
        validTiles = validTiles[::6] #Get rid of duplicates since they're already extracted
        avgProbs = np.mean(probs.reshape(-1, 6, 2), axis=1)

        confidence = [max(x) for x in avgProbs]
        for i in range(len(confidence)):
            validTiles[i]["confidence"] = confidence[i]

        isExplore = lambda x: x[0] > x[1]
        explores = np.array([isExplore(x) for x in avgProbs])
        marks = ~explores
        explores = validTiles[explores]
        marks = validTiles[marks]

        # TESTING
        # print("EXPLORES:")
        # for e in explores:
        #     print(e["tile"].row, e["tile"].col, "certainty:", e["confidence"])
        # print("MARKS:")
        # for e in marks:
        #     print(e["tile"].row, e["tile"].col, "certainty:", e["confidence"])
        # TESTING
        
        exploreMove = None
        markMove = None
        dictToList = lambda d, key: [x[key] for x in d]
        if len(explores) > 0:
            exploreMove = explores[np.argmin(dictToList(explores, "confidence"))]

        if len(marks) > 0:
            # TESTING: I'm not sure maybe this is cheating, this is waht it was before:
            # markMove = marks[np.argmax(dictToList(marks, "confidence"))]
            valid = False
            while not valid and len(marks) > 0:
                index = np.argmax(dictToList(marks, "confidence"))
                markMove = marks[index]
                surroundingTiles = self.getAllSurroundingTiles(markMove["tile"])
                surroundingValues = [t.remainingValue for t in surroundingTiles if t is not None and t.explored]
                if 0 in surroundingValues:
                    marks = np.delete(marks, index)
                    # print("-"*16, "Tried to make a stupid move :(", "-"*16, ":", "MARK:", markMove["tile"].row, markMove["tile"].col)
                    # print("Instead I will", "Explore:", markMove["tile"].row, markMove["tile"].col)
                    self.board.explore(markMove["tile"].row, markMove["tile"].col)
                    return self.board
                else:
                    valid = True

        exploredProportion = 1 #EXPLORE_COEFF * self.getExploredProportion() # TODO: Maybe more like how explored is an area?

        if exploreMove is not None and exploreMove["confidence"] > MOVE_CERTAINTY_THRESHOLD * exploredProportion: # TODO: explore more in the early game, mark late game
            move = 0
            tile = exploreMove["tile"]
        elif markMove is not None and markMove["confidence"] > MARK_CERTAINTY_THRESHOLD / exploredProportion:
            move = 1
            tile = markMove["tile"]
        else:
            # TODO: consider unexplored tiles adjacent to validTiles ?
            # print("-"*16, "I AM UNCERTAIN ABOUT THIS MOVE!", "-"*16)
            farTiles = []
            for r in self.board.board:
                for tile in r:
                    if not tile.explored and tile not in dictToList(validTiles, "tile"):
                        farTiles.append(tile)
            move = 0
            if len(farTiles) > 0:
                tile = np.random.choice(farTiles)
            else:
                tile = exploreMove["tile"]

        # print("Chosen tile:", tile.row, tile.col)
        # print("Chosen move:", "Mark" if move == 1 else "Explore")
        # print("Certainty:", markMove["confidence"] if move == 1 else exploreMove["confidence"])
        # print("Thresholds: Explore", MOVE_CERTAINTY_THRESHOLD * exploredProportion, "Mark", MARK_CERTAINTY_THRESHOLD / exploredProportion)
        if move == 1:
            self.mark(tile.row, tile.col)
        else:
            if self.board.explore(tile.row, tile.col):
                return None # Game exploded
        
        return self.board

    def getExploredProportion(self):
        numExplored = 0
        numUnexplored = 0
        for r in self.board.board:
            for t in r:
                if t.explored or t.marked:
                    numExplored += 1
                else:
                    numUnexplored += 1
        return numExplored / (numUnexplored + numExplored)

    def determineProbs(self, tiles):
        """ Returns the output of the nn on the given tiles 
            As well as a mask for the explores and marks
        """
        if MODEL == "2dnn":
            input = 12
        if MODEL == "2dnnNEW":
            input = 11
        
        if torch.cuda.is_available():
            tiles = tiles.view((-1, input, 5, 5)).cuda()
        else:
            tiles = tiles.view((-1, input, 5, 5))

        probs = self.net.classifyTiles(tiles)
        return probs
    
    @profile
    def play(self, verbose=True):
        """
        Plays the game, returns stats about the game played
        """
        self.firstMove()
        if verbose:
            print(self.board)
        totalMoves = 1
        moveTimes = []

        while not self.board.isSolved():
            t0 = perf_counter()
            boardState = self.move()
            t1 = perf_counter()
            timeToMove = t1 - t0
            totalMoves += 1
            moveTimes += [timeToMove]

            if boardState:
                if verbose:
                    print(str(boardState))
            else:
                # Game is lost
                return {
                    "win": False,
                    "explored": self.getExploredProportion(),
                    "numMoves": totalMoves,
                    "moveTimes": moveTimes,
                }
        return {
            "win": True,
            "explored": 1,
            "numMoves": totalMoves,
            "moveTimes": moveTimes,
        }
        
if __name__ == "__main__":
    nns = NeuralNetSolver(24, 24, 99)
    stats = nns.play(verbose=True)
    if stats["win"]:
        print("Hooray, the robot won!")
        print(stats)
    else:
        print("Stupid robot died :(")
        print(stats)


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
    #     data = data.cpu().numpy()
    #     mistakeData = data[mistakes]
    #     mistakeLabels = labels[mistakes]
    #     correctPredValues = output[~mistakes]
        
    #     print("First mistake:", mistakePredValues[0])
    #     printData(mistakeData[0])
    #     print("Actual value:", mistakeLabels[0])

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

        #Drop values where the difference is under threshold
        # threshold = lambda x: abs(x[0] - x[1]) > CERTAINTY_THRESHOLD
        # mask = [threshold(x) for x in output]  # TODO: loops bad
        # certainOut = output[mask]
        # certainLabels = labels[mask]
        # certainAcc = np.mean(certainLabels == np.argmax(certainOut, axis=1))
        # print("Thresholded accuracy:", certainAcc)
        # print("Number certain results:", certainLabels.shape)
        # print("Percentage certain results:", certainLabels.shape[0] / BATCH_SIZE)