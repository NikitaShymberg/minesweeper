# NOTE: I wonder how bomb density will affect this stuff

import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from game.board import Board
from game.tile import Tile
from game.constants import MAX_BOMBS, BOMB, WIDTH, HEIGHT

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