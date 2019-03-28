import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from game.board import Board
from game.tile import Tile
from game.constants import MAX_BOMBS, BOMB, WIDTH, HEIGHT
from itertools import permutations, chain
from sympy.utilities.iterables import multiset_permutations
from math import factorial
import numpy as np

#TESTING
from pprint import pprint

class BruteForceSolver:
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
    
    def firstMove(self):
        # http://www.minesweeper.info/wiki/Strategy#First_Click
        self.board.explore(0, 0)
    
    def mark(self, row, col):
        """ Marks the tile at the row and column as a bomb and updates the remaining value of surrounding tiles """
        self.board.mark(row, col)
        self.unmarkedBombs -= 1
        tilesToReduce = self.getAllSurroundingTiles(Tile(0, row, col))
        for t in tilesToReduce:
            if t is not None and t.remainingValue != 0 and t.value != BOMB:
                self.board.board[t.row][t.col].remainingValue -= 1

    def move(self):
        """
        Performs the optimal move, if the move was successful returns the
        current board, else returns None to signify that a mine was triggered
        """
        # print("Number of bombs left:", self.unmarkedBombs)
        tilesToConsider = self.getTilesAdjacentToExploredTiles()
        probabilityBoard = self.calculateProbabilities(tilesToConsider)
        completedMove = False # Make all the certain moves at once

        # Check if there is any certain bombs
        for i, row in enumerate(probabilityBoard):
            for j, pBomb in enumerate(row):
                if pBomb == 1 and not self.board.board[i][j].marked:
                    # print("MOVE: mark", i, j)
                    self.mark(i, j)
                    completedMove = True

        # Explore any certain safe spaces
        for i, row in enumerate(probabilityBoard):
            for j, pBomb in enumerate(row):
                if pBomb == 0:
                    # print("MOVE: explore1", i, j)
                    if self.board.explore(i, j):
                        return None # Game exploded
                    completedMove = True
        
        if completedMove:
            return self.board            

        # Explore the most likely safe space
        # TODO: speed me up scotty
        minimum = 1
        mini = -1
        minj = -1
        for i, row in enumerate(probabilityBoard):
            for j, pBomb in enumerate(row):
                if pBomb < minimum:
                    minimum = pBomb
                    mini = i
                    minj = j
        # print("MOVE: explore2", i, j)
        if self.board.explore(mini, minj):
            return None # Game exploded
        
        return self.board
                        
    def calculateProbabilities(self, tilesToConsider):
        """ Calculates the probability of each tile being a bomb TODO: too long func """
        numUnexplored = self.countUnexploredTiles()
        noInfoTiles = numUnexplored - len(tilesToConsider)

        # print("Number of tiles to consider:", len(tilesToConsider))
        # print("Total unexplored tiles:", numUnexplored)
        # print("Tiles with no info:", noInfoTiles)

        possibleBombs = self.permuteBombsInTiles(tilesToConsider)
        exploredTiles = [tile for row in self.board.board for tile in row if tile.explored] 

        # Format: [permutation[tile and isBomb]]
        validBombs = [x for x in possibleBombs if self.isPermutationValid(x, exploredTiles, noInfoTiles)]

        # Calculate probability of having that many bombs in tilesToConsider
        bombCounts = [] # bombCounts[i] = the number of bombs in validBombs[i]
        for v in validBombs:
            bombCounts.append(len([isBomb for isBomb in v if isBomb['isBomb'] == BOMB]))
        
        # The number of bomb permutations given bombs in tilesToConsider
        permutationsOfOtherTiles = [self.countPermutations(noInfoTiles, self.unmarkedBombs - bc) for bc in bombCounts]
        # print("Permutations of other tiles", permutationsOfOtherTiles) # TODO: testme?
        
        # print("Number of bombs in each permutation:", bombCounts)

        isBombCount = [0] * len(tilesToConsider)
        for i, tile in enumerate(tilesToConsider):
            for j, permutation in enumerate(validBombs):
                for p in permutation:
                    
                    # TESTING
                    # if i == 0:
                    #     print(p['tile'].row, p['tile'].col, p['isBomb'])
                    
                    if p['tile'] == tile and p['isBomb']:
                        isBombCount[i] += permutationsOfOtherTiles[j]
                # TESTING:
                # if i == 0:
                #     print('-'*32)
        # print("Number of times this tile was a bomb:", isBombCount)
        if sum(permutationsOfOtherTiles) != 0:
            isBombProbability = [count / sum(permutationsOfOtherTiles) for count in isBombCount]
        else:
            isBombProbability = [0] * len(isBombCount)
        # print("Probability of this tile being a bomb:", isBombProbability)
        
        if noInfoTiles != 0:
            if len(bombCounts) != 0:
                noInfoBombChance = (self.unmarkedBombs - sum(bombCounts)/len(bombCounts)) / noInfoTiles
            else:
                noInfoBombChance = self.unmarkedBombs / noInfoTiles
        else:
            noInfoBombChance = 0
        # print("Chance of having a bomb in no info tile:", noInfoBombChance)

        tilesAndProbability = zip(tilesToConsider, isBombProbability)
        
        # Build the probability board
        probabilityBoard = [ [ None for i in range(WIDTH) ] for j in range(HEIGHT) ]

        for tp in tilesAndProbability:
            probabilityBoard[tp[0].row][tp[0].col] = tp[1]

        for tile in exploredTiles:
            probabilityBoard[tile.row][tile.col] = 2 # don't click me, I', explored
        
        for i, row in enumerate(probabilityBoard):
            for j, tile in enumerate(row):
                if tile is None:
                    probabilityBoard[i][j] = noInfoBombChance
                if self.board.board[i][j].marked:
                    probabilityBoard[i][j] = 3 # don't click me, I'm marked

        # TESTING
        # print("\n Probability board:")
        # for q in probabilityBoard:
        #     for p in q:
        #         print("{0:.2f}".format(p), end=" ")
        #     print()

        return probabilityBoard
    
    def countPermutations(self, n, r):
        # HACK: r should never be > n but sometimes it is :(
        if n - r < 0:
            return 0
        else:
            return factorial(n) / (factorial(n - r))
    
    
    def isPermutationValid(self, permutation, explored, noInfoTiles):
        """ Returns true if the given permutation of bombs is valid given the explored tiles """
        # If the number of bombs for outside the permutation > noInfoTiles then invalid
        if self.unmarkedBombs + sum([p['isBomb'] for p in permutation]) > noInfoTiles:
            return False

        for e in explored:
            adjacentBombs = [p for p in permutation if self.isAdjacent(e, p['tile']) and p['isBomb'] == BOMB]
            # if len(adjacentBombs) > 0 and len(adjacentBombs) ==  e.value - 1:
            #     print("ERROR?:", len(adjacentBombs), e.row, e.col, e.value)
            if len(adjacentBombs) != e.remainingValue:
                return False
       
        return True
    
    def isAdjacent(self, a, b):
        """ Returns true if the given tiles are adjacent TODO: maybe smartify? """
        if a.row == b.row - 1:
            if a.col == b.col - 1 or a.col == b.col or a.col == b.col + 1:
                return True
            return False
        if a.row == b.row:
            if a.col == b.col - 1 or a.col == b.col + 1:
                return True
            return False
        if a.row == b.row + 1:
            if a.col == b.col - 1 or a.col == b.col or a.col == b.col + 1:
                return True
            return False
    
    def permuteBombsInTiles(self, tiles):
        """ 
        Set up all the possible (valid and invalid) permutations of bombs in the given tiles 
        Returns a list of dicts with {tile: $tile, isBomb: $boolean}
        """
        possiblePermutations = []
        # FIXME: this is also slow
        for numBombs in range(min(self.unmarkedBombs + 1, len(tiles))):
            bombArrangement = [BOMB] * numBombs + [0] * (len(tiles) - numBombs)
            possiblePermutations += list(multiset_permutations(bombArrangement))

        # Format: [bomb arrangement[isBomb]]
        possiblePermutations = np.array(possiblePermutations, dtype=int)
            
        # TESTING
        # print("Bomb layouts:")
        # for pp in possiblePermutations:
        #     pprint(list(pp))

        tiles = np.array(list(tiles)).reshape(1, -1)
        # FIXME: I'm STILL slow and I eat memory like popcorn
        mapBombsToTiles = lambda bombs, tile: {"tile": tile, "isBomb": bombs}
        mapBombsToTiles = np.vectorize(mapBombsToTiles)
        # print("PossiblePermutations shape:", possiblePermutations.shape) # TESTING - number of perms and tiles
        mappedPermutations = mapBombsToTiles(possiblePermutations, tiles)
        
        return mappedPermutations

    
    def getTilesAdjacentToExploredTiles(self):
        """ Returns all unique unexplored tiles that are next to an explored tile """
        exploredTiles = [tile for row in self.board.board for tile in row if tile.explored]
        tilesToConsider = [self.getSurroundingTiles(tile) for tile in exploredTiles]
        tilesToConsider = set(chain.from_iterable(tilesToConsider))

        # TESTING
        # for t in tilesToConsider:
        #     print("Considering: ", t.row, t.col)

        return tilesToConsider

    def getAllSurroundingTiles(self, tile):
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

    def getSurroundingTiles(self, tile, explored=False):
        surrounding = self.getAllSurroundingTiles(tile)
        
        if not explored:
            filteredTiles = lambda x: not x.explored and not x.marked if x is not None else False
        else:
            filteredTiles = lambda x: x.explored if x is not None else False

        surrounding = list(filter(filteredTiles, surrounding))
        
        return surrounding
    
    def countUnexploredTiles(self):
        # TODO: speed me up scotty?
        count = 0
        for row in self.board.board:
            for tile in row:
                if not tile.explored:
                    count += 1
        
        return count

    def play(self, verbose=True):
        """
        Plays the game, returns stats about the game played
        """
        self.firstMove()
        if verbose:
            print(self.board)

        while not self.board.isSolved():
            boardState = self.move()
            if boardState:
                if verbose:
                    print(str(boardState))
            else:
                # Game is lost
                return False
        return True

if __name__ == "__main__":
    bfs = BruteForceSolver()
    if bfs.play(verbose=True):
        print("Hooray, the robot won!")
    else:
        print("Stupid robot died :(")
    