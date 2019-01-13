import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

# from game.constants import *
from game.board import Board
from game.tile import Tile
from game.constants import MAX_BOMBS, BOMB, WIDTH, HEIGHT
from itertools import permutations, chain
from sympy.utilities.iterables import multiset_permutations
from math import factorial

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
        # TODO: smartify and cite
        self.board.explore(0,0)

    def move(self):
        print("Number of bombs left:", self.unmarkedBombs)
        probabilityBoard = self.calculateProbabilities()
        
        return probabilityBoard
    
    def calculateProbabilities(self):
        """ Calculates the probability of each tile being a bomb TODO: too long func """
        tilesToConsider = self.getTilesAdjacentToExploredTiles()
        numUnexplored = self.countUnexploredTiles()
        noInfoTiles = numUnexplored - len(tilesToConsider)

        print("Number of tiles to consider:", len(tilesToConsider))
        print("Total unexplored tiles:", numUnexplored)
        print("Tiles with no info:", noInfoTiles)

        possibleBombs = self.permuteBombsInTiles(tilesToConsider)
        exploredTiles = [tile for row in self.board.board for tile in row if tile.explored] 

        # Format: [permutation[tile and isBomb]]
        validBombs = [x for x in possibleBombs if self.isPermutationValid(x, exploredTiles)]

        # Calculate probability of having that many bombs in tilesToConsider
        bombCounts = [] # bombCounts[i] = the number of bombs in validBombs[i]
        for v in validBombs:
            bombCounts.append(len([isBomb for isBomb in v if isBomb['isBomb'] == BOMB]))
        
        # The number of bomb permutations given bombs in tilesToConsider
        permutationsOfOtherTiles = [self.countPermutations(noInfoTiles, self.unmarkedBombs - bc) for bc in bombCounts]
        print("Permutations of other tiles", permutationsOfOtherTiles) # TODO: testme?
        
        print("Number of bombs in each permutation:", bombCounts)

        isBombCount = [0] * len(tilesToConsider)
        for i, tile in enumerate(tilesToConsider):
            for j, permutation in enumerate(validBombs):
                for p in permutation:
                    
                    # TESTING
                    if i == 0:
                        print(p['tile'].row, p['tile'].col, p['isBomb'])
                    
                    if p['tile'] == tile and p['isBomb']:
                        isBombCount[i] += permutationsOfOtherTiles[j] # NOTE I think this is where you weigh it?
                        # Add the number of times that this permutation happened in ALL possible perms
                # TESTING:
                if i == 0:
                    print('-'*32)
        print("Number of times this tile was a bomb:", isBombCount)
        isBombProbability = [count / sum(permutationsOfOtherTiles) for count in isBombCount] # NOTE and here divide by TOTAL possible perms
        print("Probability of this tile being a bomb:", isBombProbability)
        
        noInfoBombChance = (self.unmarkedBombs - sum(bombCounts)/len(bombCounts)) / noInfoTiles
        print("Chance of having a bomb in no info tile:", noInfoBombChance)

        tilesAndProbability = zip(tilesToConsider, isBombProbability)
        
        # Build the probability board
        probabilityBoard = [ [ None for i in range(WIDTH) ] for j in range(HEIGHT) ]

        for tp in tilesAndProbability:
            probabilityBoard[tp[0].row][tp[0].col] = tp[1]

        for tile in exploredTiles:
            probabilityBoard[tile.row][tile.col] = -1 # don't click me
        
        for i, row in enumerate(probabilityBoard):
            for j, tile in enumerate(row):
                if tile is None:
                    probabilityBoard[i][j] = noInfoBombChance

        for q in probabilityBoard:
            print(q)

        return probabilityBoard
    
    def countPermutations(self, n, r):
        return factorial(n) / (factorial(n - r))
    
    def isPermutationValid(self, permutation, explored):
        """ Returns true if the given permutation of bombs is valid given the explored tiles """
        for e in explored:
            adjacentBombs = [p for p in permutation if self.isAdjacent(e, p['tile']) and p['isBomb'] == BOMB]
            if len(adjacentBombs) != e.value:
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
        for numBombs in range(min(self.unmarkedBombs + 1, len(tiles) + 1)):
            bombArrangement = [BOMB] * numBombs + [0] * (len(tiles) - numBombs)
            possiblePermutations.append(multiset_permutations(bombArrangement)) #TODO: don't need the empty set (it's fine?)

        # TODO: timeit, if the other thing is faster, apply to all
        # Format: [number of bombs[bomb arrangement[isBomb]]]
        possiblePermutations = set(possiblePermutations)
        # for p in possiblePermutations:
        #     if p not in q:
        #         q.add(p)
            
        # TESTING
        # print("Bomb layouts:")
        # for pp in possiblePermutations:
        #     pprint(list(pp))

        tiles = list(tiles)
        mappedPermutations = []
        for pp in possiblePermutations:
            for permutation in pp:
                arrangementGroup = []
                for i, isBomb in enumerate(permutation):
                    arrangementGroup.append({
                        "tile": tiles[i],
                        "isBomb": isBomb
                    })
                mappedPermutations.append(arrangementGroup)
                
        return mappedPermutations

    def getTilesAdjacentToExploredTiles(self):
        """ Returns all unique unexplored tiles that are next to an explored tile """
        exploredTiles = [tile for row in self.board.board for tile in row if tile.explored]
        tilesToConsider = [self.getSurroundingTiles(tile) for tile in exploredTiles]
        tilesToConsider = set(chain.from_iterable(tilesToConsider))

        return tilesToConsider

    def getSurroundingTiles(self, tile, explored=False):
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
        
        if not explored:
            filteredTiles = lambda x: not x.explored if x is not None else False
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

if __name__ == "__main__":
    bfs = BruteForceSolver()
    bfs.firstMove()
    print(bfs.board)

    # pprint(bfs.move())
    bfs.move()
