# TODO: refactor a little
import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from game.board import Board
from game.tile import Tile
from game.constants import MAX_BOMBS, BOMB, WIDTH, HEIGHT, TRAINING_DATA_FILE
from bruteForce.bruteForce import BruteForceSolver # TODO: that's kidna gross 
import h5py
import numpy as np
from random import randrange

def getAllSurroundingTiles(board, tile):
    """ Returns a list of the 5x5 grid of surrounding tiles any invalid tile is None """
    # NOTE: surrounding format = [left, right, up, down, 
    #                             upLeft, leftLeft, upLeftLeft, upUpLeftLeft, upUpLeft, upUp
    #                             upRight, upUpRight, upUpRightRight, upRightRight
    #                             downLeft, downLeftLeft, downDownLeftLeft, downDownLeft
    #                             downRight, downDown, downDownRight, downDownRightRight, downRightRight, righRight]
    surrounding = []

    row = tile.row
    col = tile.col

    surrounding.append(board.left(row, col))
    surrounding.append(board.right(row, col))
    surrounding.append(board.up(row, col))
    surrounding.append(board.down(row, col))

    # Process 5 surrounding tiles of upLeft
    ul = board.upLeft(row, col)
    surrounding.append(ul)
    if ul is None:
        for _ in range(5):
            surrounding.append(None)
    else:
        surrounding.append(board.downLeft(ul.row, ul.col))
        surrounding.append(board.left(ul.row, ul.col))
        surrounding.append(board.upLeft(ul.row, ul.col))
        surrounding.append(board.up(ul.row, ul.col))
        surrounding.append(board.upRight(ul.row, ul.col))

    # Process 3 surrounding tiles of upRight
    ur = board.upRight(row, col)
    surrounding.append(ur)
    if ur is None:
        for _ in range(3):
            surrounding.append(None)
    else:
        surrounding.append(board.up(ur.row, ur.col))
        surrounding.append(board.upRight(ur.row, ur.col))
        surrounding.append(board.right(ur.row, ur.col))
    
    # Process 3 surrounding tiles of downLeft 
    dl = board.downLeft(row, col)
    surrounding.append(dl)
    if dl is None:
        for _ in range(3):
            surrounding.append(None)
    else:
        surrounding.append(board.left(dl.row, dl.col))
        surrounding.append(board.downLeft(dl.row, dl.col))
        surrounding.append(board.down(dl.row, dl.col))

    # Process 5 surrounding tiles of downRight
    dr = board.downRight(row, col)
    surrounding.append(dr)
    if dr is None:
        for _ in range(5):
            surrounding.append(None)
    else:
        surrounding.append(board.downLeft(dr.row, dr.col))
        surrounding.append(board.down(dr.row, dr.col))
        surrounding.append(board.downRight(dr.row, dr.col))
        surrounding.append(board.right(dr.row, dr.col))
        surrounding.append(board.upRight(dr.row, dr.col))
    
    
    return surrounding

# Labels: 0 --> safe, 1 --> bomb
# TODO: TESTME!
def processTile(board, tile):
    """
    Returns a list of the values of the 5x5 grid of surrounding tiles excluding the cetre tile
    As well as the label for the centre tile
    One hot format:
    [
        [isTile, isUnexplored, isBomb, 0, 1, 2, 3, 4, 5, 6, 7, 8], ...
    ]
    """
    label = 1 if tile.value == BOMB else 0
    surroundingTiles = getAllSurroundingTiles(board, tile)
    
    values = np.zeros((24, 11))
    for i, tile in enumerate(surroundingTiles):
        if tile is None:
            values[i][0] = 1
        elif not tile.explored:
            values[i][1] = 1
        elif tile.value == BOMB:
            values[i][2] = 1
        else:
            values[i][tile.value + 3] = 1

    return values, label

def exploreSafeTile(board):
    """ Explores a tile that is not a bomb """
    foundSafeTile = False
    while not foundSafeTile:
        row = randrange(len(board.board))
        col = randrange(len(board.board[row]))
        if board.board[row][col].value != BOMB:
            foundSafeTile = True
    
    board.explore(row, col)

def generateTrainingData():
    """ Returns a one hot encoding of the data and a list of the labels """
    board = Board()
    allTileInfo = []
    allLabels = []

    # TODO: fiddle with number
    for _ in range(25):
        exploreSafeTile(board)
    
    print(str(board))
    # FIXME this is hideous
    bfs = BruteForceSolver()
    bfs.board = board
    tilesToConsider = bfs.getTilesAdjacentToExploredTiles()
    print(len(tilesToConsider))

    for tile in tilesToConsider:
        tileInfo, label = processTile(board, tile)
        allTileInfo.append(tileInfo) # FIXME append bad
        allLabels.append(label)
    
    print(allTileInfo[0])
    print(allLabels[0])

    with h5py.File(TRAINING_DATA_FILE, "w") as f:
        f["data"] = allTileInfo
        f["class"] = allLabels
    
    return allTileInfo, allLabels

if __name__ == "__main__":
    generateTrainingData()