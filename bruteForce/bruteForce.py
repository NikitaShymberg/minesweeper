import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

# from game.constants import *
from game.board import Board
from game.tile import Tile
from game.constants import MAX_BOMBS, BOMB, WIDTH, HEIGHT
from itertools import permutations, chain
from sympy.utilities.iterables import multiset_permutations

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
        # TODO: smart and cite
        self.board.explore(0,0)

    def move(self):
        print("Number of bombs left:", self.unmarkedBombs)
        probabilityBoard = self.calculateProbabilities()
        
        return probabilityBoard
    
    def calculateProbabilities(self):
        """ Calculates the probability of each tile being a bomb """
        tilesToConsider = self.getTilesAdjacentToExploredTiles()
        totalUnexplored = self.countUnexploredTiles()

        print("Number of tiles to consider:", len(tilesToConsider))
        print("Total unexplored tiles:", totalUnexplored)

        possibleBombs = self.permuteBombsInTiles(tilesToConsider)
        exploredTiles = [tile for row in self.board.board for tile in row if tile.explored] 

        validBombs = [x for x in possibleBombs if self.isPermutationValid(x, exploredTiles)]
        for v in validBombs:
            for q in v:
                print(q['isBomb'], q['tile'].row, q['tile'].col)
            print('-'*32)

        return validBombs
    
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
        # TODO: speed me up scotty
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
