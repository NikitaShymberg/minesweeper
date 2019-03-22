# TODO: refactor a little
import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from game.board import Board
from game.tile import Tile
from game.constants import *
from bruteForce.bruteForce import BruteForceSolver # TODO: that's kidna gross 
import torch
import numpy as np
from random import randrange
from itertools import permutations


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
    
    if MODEL == "nn":
    # Re-order stuff for clarity and so conv layer works good
        surrounding = [
            surrounding[7], surrounding[8], surrounding[9], surrounding[11], surrounding[12],
            surrounding[6], surrounding[4], surrounding[2], surrounding[10], surrounding[13],
            surrounding[5], surrounding[0], surrounding[1], surrounding[23],
            surrounding[15], surrounding[14], surrounding[3], surrounding[18], surrounding[22],
            surrounding[16], surrounding[17], surrounding[19], surrounding[20], surrounding[21]
            ]
    elif MODEL == "2dnn" or MODEL == "2dnnNEW":
        surrounding = [
            [surrounding[7], surrounding[8], surrounding[9], surrounding[11], surrounding[12],],
            [surrounding[6], surrounding[4], surrounding[2], surrounding[10], surrounding[13],],
            [surrounding[5], surrounding[0], Tile(0, -1, -1), surrounding[1], surrounding[23],],
            [surrounding[15], surrounding[14], surrounding[3], surrounding[18], surrounding[22],],
            [surrounding[16], surrounding[17], surrounding[19], surrounding[20], surrounding[21]],
            ]
    
    return surrounding

# Labels: 0 --> safe, 1 --> bomb
def processTile(board, tile, mode="train"):
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
    
    if MODEL == "nn":
        values = np.zeros((24, 12))
        for i, curTile in enumerate(surroundingTiles):
            if curTile is None or curTile.marked:
                values[i][0] = 1
            elif not curTile.explored:
                values[i][1] = 1
            elif curTile.value == BOMB:
                values[i][2] = 1
            else:
                if mode == "train":
                    values[i][tile.value + 3] = 1
                elif mode == "play":
                    values[i][tile.remainingValue + 3] = 1

    elif MODEL == "2dnn":
        values = np.zeros((5, 5, 12))
        for i, row in enumerate(surroundingTiles):
            for j, curTile in enumerate(row):
                if curTile is None or curTile.marked:
                    values[i][j][0] = 1
                elif not curTile.explored:
                    values[i][j][1] = 1
                elif curTile.value == BOMB:
                    print("I SAW A BOMB?????", "*"*32) #TESTING please never print this
                    values[i][j][2] = 1
                else:
                    if mode == "train":
                        values[i][j][curTile.value + 3] = 1
                    elif mode == "play":
                        # print("remaining value:", tile.remainingValue, "at", tile.row, tile.col)
                        values[i][j][tile.remainingValue + 3] = 1
    
    # One hot format:
    # [
    #     [isTile, isUnexplored, 0, 1, 2, 3, 4, 5, 6, 7, 8], ...
    # ]
    elif MODEL == "2dnnNEW":
        values = np.zeros((5, 5, 11))
        for i, row in enumerate(surroundingTiles):
            for j, curTile in enumerate(row):
                if curTile is None or curTile.marked:
                    values[i][j][0] = 1
                elif not curTile.explored:
                    values[i][j][1] = 1
                else:
                    if mode == "train":
                        values[i][j][curTile.value + 2] = 1
                    elif mode == "play":
                        values[i][j][tile.remainingValue + 2] = 1


    return values, label

def printData(tiles):
    # tiles = tiles.cpu().numpy()
    tiles = tiles.reshape(5, 5, 11)
    for r in tiles:
        for t in r:
            if t[0] == 1:
                val = " X"
            elif t[1] == 1:
                val = " ?"
            elif t[2] == 1:
                val = " 0"
            elif t[3] == 1:
                val = " 1"
            elif t[4] == 1:
                val = " 2"
            elif t[5] == 1:
                val = " 3"
            elif t[6] == 1:
                val = " 4"
            elif t[7] == 1:
                val = " 5"
            elif t[8] == 1:
                val = " 6"
            elif t[9] == 1:
                val = " 7"
            elif t[10] == 1:
                val = " 8"
            print(val, end=" ")
        print()

def transformBoard(board):
    """ Given a board, transforms all tiles next to eplored ones into a format that the nn can take
    Returns a list of dicts with the "tile" and "nn" keys
    """
    # FIXME this is hideous
    out = []
    bfs = BruteForceSolver()
    bfs.board = board
    tilesToConsider = bfs.getTilesAdjacentToExploredTiles()
    for tile in tilesToConsider:
        processedTile, l = processTile(board, tile, mode="play")
        processTiles, _ = modifyTiles(processedTile, l) # TESTING ?
        for pt in processTiles:
            out.append({
                "tile": tile,
                "nn": pt
            })
    
    return np.array(out)


def exploreSafeTile(board):
    """ Explores a tile that is not a bomb """
    foundSafeTile = False
    i = 0
    while not foundSafeTile and i < 50:
        i += 1
        row = randrange(len(board.board))
        col = randrange(len(board.board[row]))
        if board.board[row][col].value != BOMB and not board.board[row][col].explored:
            foundSafeTile = True
    
    if i < 50:
        board.explore(row, col)

def modifyTiles(tileInfo, label):
    """ Returns a list of rotated and flipped tiles and their labels """
    allTileInfo = []
    allLabels = []

    allTileInfo.append(tileInfo)
    allTileInfo.append(np.fliplr(tileInfo))
    allTileInfo.append(np.flipud(tileInfo))
    allTileInfo.append(np.rot90(tileInfo, 1, axes=(0,1)))
    allTileInfo.append(np.rot90(tileInfo, 2, axes=(0,1)))
    allTileInfo.append(np.rot90(tileInfo, 3, axes=(0,1)))
    for _ in range(6):
        allLabels.append(label)

    return allTileInfo, allLabels

def generateTrainingData():
    """ Returns BATCH_SIZE samples a one hot encoding of the data and a list of the labels """
    allTileInfo = []
    allLabels = []
    numMoves = 0

    while len(allLabels) < 2.5 * BATCH_SIZE:
        numMoves = (numMoves % 16) + 1
        board = Board()
        for _ in range(numMoves):
            exploreSafeTile(board)
        
        # FIXME this is hideous
        bfs = BruteForceSolver()
        bfs.board = board
        tilesToConsider = bfs.getTilesAdjacentToExploredTiles()
        if len(tilesToConsider) > 13:
            tilesToConsider = list(tilesToConsider)[:13] # Otherwise it takes too long...
        probs = bfs.calculateProbabilities()
        tilesToConsider = filterBadTiles(tilesToConsider, board, probs)

        for tile in tilesToConsider: # TODO TEST ME!!!
            tileInfo, label = processTile(board, tile)
            modifiedTiles, modifiedLabels = modifyTiles(tileInfo, label)
            allTileInfo += modifiedTiles
            allLabels += modifiedLabels
            
    
    allTileInfo, allLabels = balanceLabels(allTileInfo, allLabels)
    if MODEL == "nn":
        allTileInfo = allTileInfo.reshape(BATCH_SIZE, 12, 24)
    elif MODEL == "2dnn":
        allTileInfo = allTileInfo.reshape(BATCH_SIZE, 12, 5, 5)
    elif MODEL == "2dnnNEW":
        allTileInfo = allTileInfo.reshape(BATCH_SIZE, 11, 5, 5)

    allTileInfo = torch.from_numpy(allTileInfo).float()
    allLabels = torch.from_numpy(allLabels)
    if torch.cuda.is_available():
        allTileInfo = allTileInfo.cuda()
        allLabels = allLabels.cuda()

    return allTileInfo, allLabels

def filterBadTiles(tiles, board, probs):
    """ Removes tiles that are very hard to classify """
    # print("\n Probability board:") #TESTING
    # for q in probs:
    #     for p in q:
    #         print("{0:.2f}".format(p), end=" ")
    #     print()

    goodTiles = []
    for t in tiles:
        #toKick = abs(probs[t.row][t.col] - .5) # Highest chance of kicking 50%, lowest at the edges
        #if toKick > 0.35 and np.random.rand() / 2 < toKick: # But definitely kick the real bad ones
        if probs[t.row][t.col] == 0 or probs[t.row][t.col] == 1:
            # print("GOOD TILE AT:", t.row, t.col, "Value:", probs[t.row][t.col]) TESTING
            goodTiles.append(t)
    
    return goodTiles

def balanceLabels(allTileInfo, allLabels):
    """ Returns balanced data - i.e. equal amounts of both classes """
    allTileInfo = np.asarray(allTileInfo)
    allLabels = np.asarray(allLabels)

    mask0 = np.random.choice(np.where(allLabels == 0)[0], BATCH_SIZE // 2, replace=False)
    mask1 = np.random.choice(np.where(allLabels == 1)[0], BATCH_SIZE // 2, replace=False)
    allTileInfo = np.vstack((allTileInfo[mask0], allTileInfo[mask1])) 
    allLabels = np.hstack((allLabels[mask0], allLabels[mask1]))

    mask = np.random.permutation(np.arange(BATCH_SIZE))
    allTileInfo = allTileInfo[mask]
    allLabels = allLabels[mask]
    
    return allTileInfo, allLabels

if __name__ == "__main__":
    d, l = generateTrainingData()
    print("L:", l.cpu().numpy())
    print("Num Bomb labels:", np.sum(l.cpu().numpy()))
