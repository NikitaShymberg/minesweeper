import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from game.board import Board
from game.tile import Tile
from game.constants import *
from bruteForce.bruteForce import BruteForceSolver
#from neuralNet.twoD_nn import miniNet
from psutil import cpu_percent

def test_bfs():
    for i, boardLayout in enumerate(BOARDS):
        print("Playing on board:", boardLayout)
        for gameNum in range(NUM_GAMES[i]):
            print("Game number:", gameNum)
            bfs = BruteForceSolver(boardLayout["rows"], boardLayout["cols"], boardLayout["numBombs"])
            cpu_percent(percpu=True)
            stats = bfs.play(verbose=False)
            cpu = cpu_percent(percpu=True)
            gameStats = {
                "boardLayout": boardLayout, # The type of board that the game was played on
                "moveTimes": stats["moveTimes"], # The time taken to complete each move
                "numMoves": stats["numMoves"], # The total number of moves that happened in the game
                "win": stats["win"], # Whether the robot won or lost
                "explored": stats["explored"], # The explored proportion of the board
                "CPU": cpu, # Average CPU usage during the game
            }
            with open(STATS_FILE + "bfs", "a") as f:
                f.write(str(gameStats))

# TODO: ensure nothing is being printed in each AI
def test_nn():
    pass

def test_q():
    pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Please pick a valid model to test")
    if sys.argv[1] == "bfs":
        test_bfs()
    elif sys.argv[1] == "nn":
        test_nn()
    elif sys.argv[1] == "q":
        test_q()
    else:
        raise Exception("Unknown model type")
