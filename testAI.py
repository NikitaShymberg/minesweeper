import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from game.board import Board
from game.tile import Tile
from game.constants import *
from bruteForce.bruteForce import BruteForceSolver
from neuralNet.twoD_nn import miniNet

# TODO: ensure nothing is being printed in each AI
# TODO: Things to collect:
    # Time taken per move
    # Number of moves in a game
    # Win or loss
    # Percentage of board explored
    # Memory usage
    # CPU/GPU usage

def test_bfs():
    gameStats = []
    for boardLayout in BOARDS:
        for gameNum in NUM_GAMES:
            bfs = BruteForceSolver()
            while not bfs.board.isSolved():
                bfs.move() # TODO: record time properly here
            gameStats.append({
                "boardLayout": boardLayout,
                "moveTime": None,
                "NumMoves": None,
                "win": None,
                "AvgMem": None,
                "MaxMem": None,
                "CPU": None,
            })

def test_nn():
    pass

def test_q():
    pass

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Please pick a valid model to test")
    if sys.argv[1] == "bfs":
        test_bfs()
    elif sys.argv[1] == "nn":
        test_nn()
    elif sys.argv[1] == "q":
        test_q()
    else:
        raise Exception("Unknown model type")