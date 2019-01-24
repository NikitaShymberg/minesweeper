# TODO: refactor a little
import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from game.board import Board
from game.tile import Tile
from game.constants import MAX_BOMBS, BOMB, WIDTH, HEIGHT, TRAINING_DATA_FILE
import h5py
import numpy as np

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
        elif not tile.explored: #TODO: explore stuff
            values[i][1] = 1
        elif tile.value == BOMB:
            values[i][2] = 1
        else:
            values[i][tile.value + 3] = 1

    return values, label

def generateTrainingData():
    board = Board()
    allTileInfo = []
    allLabels = []

    # TODO: explore some stuff here

    for row in board.board:
        for tile in row:
            tileInfo, label = processTile(board, tile)
            allTileInfo.append(tileInfo) # FIXME append bad
            allLabels.append(label)
        
    with h5py.File(TRAINING_DATA_FILE, "w") as f:
        f["data"] = allTileInfo
        f["class"] = allLabels

if __name__ == "__main__":
    generateTrainingData()