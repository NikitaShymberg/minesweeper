from board import Board
from constants import BOARDS, NUM_GAMES, STATS_FILE
from random import randint


for i, boardLayout in enumerate(BOARDS):
        print("Playing on board:", boardLayout)
        for gameNum in range(NUM_GAMES[i]):
            print("Game number:", gameNum)
            board = Board(boardLayout["rows"], boardLayout["cols"], boardLayout["numBombs"])

            win = True
            numMoves = 0
            while not board.isSolved(): # GAME
                row, col = 0, 0
                while(board.board[row][col].explored): # FINDMOVE
                    row = randint(0, boardLayout["rows"] - 1)
                    col = randint(0, boardLayout["cols"] - 1)
                if board.explore(row, col): # We blew up
                    win = False
                    break
                numMoves += 1
            
            count = 0
            for row in board.board:
                for tile in row:
                    if tile.explored:
                        count += 1
            exploredProportion = count / (boardLayout["rows"] * boardLayout["cols"])
            gameStats = {
                "boardLayout": boardLayout, # The type of board that the game was played on
                "moveTimes": [0], # The time taken to complete each move
                "numMoves": numMoves, # The total number of moves that happened in the game
                "win": win, # Whether the robot won or lost
                "explored": exploredProportion, # The explored proportion of the board
                "CPU": 0, # Average CPU usage during the game
            }
            with open(STATS_FILE + "rand", "a") as f:
                f.write(str(gameStats) + ", ")