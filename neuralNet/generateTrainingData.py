# TODO: refactor a little
import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from game.board import Board
from game.tile import Tile
from game.constants import MAX_BOMBS, BOMB, WIDTH, HEIGHT
import h5py

def getAllSurroundingTiles(board, tile):
    """ Returns a list of the 5x5 grid of surrounding tiles any invalid tile is None """ 
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
    """
    label = 1 if tile.value == BOMB else 0
    surroundingTiles = getAllSurroundingTiles(board, tile)
    
    values = []
    for tile in surroundingTiles:
        if tile is not None:
            values.append(tile.value / 8)
        else:
            values.append(-1) # Not a tile

    return values, label

board = Board()
with h5py.File("trainingData.h5", "w") as f:
    allTileInfo = []
    allLabels = []

    for row in board.board:
        for tile in row:
            tileInfo, label = processTile(board, tile)
            allTileInfo.append(tileInfo)
            allLabels.append(label)
    
    f["data"] = allTileInfo
    f["class"] = allLabels