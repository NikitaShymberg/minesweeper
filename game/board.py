import sys
sys.path.append('.') #TODO: must be nicer
sys.path.append('..') #TODO: must be nicer

from game.constants import WIDTH, HEIGHT, BOMB, MAX_BOMBS
from random import randint
from game.tile import Tile

class Board:

    def __init__(self):
        self.firstMove = True
        self.correctMarks = 0
        self.board = [[Tile(0, r, c) for c in range(WIDTH)] for r in range(HEIGHT)]
        self.setupBoard() # FIXME: REMOVEME

    def setupBoard(self):
        for r in range(HEIGHT):
            for c in range(WIDTH):
                self.board[r][c].value = 0
        setBombs = 0
        # NOTE: might be slow if bomb dense
        while(setBombs < MAX_BOMBS):
            row = randint(0, WIDTH-1)
            col = randint(0, HEIGHT-1)
            if(self.board[row][col].value != BOMB):
                self.board[row][col] = Tile(BOMB, row, col)
                setBombs += 1

        # TODO: improve?
        for r in range(WIDTH):
            for c in range(HEIGHT):
                self.updateCounts(r, c)

    def updateCounts(self, row, col):
        """ Given a cell, update its value """
        if(self.board[row][col].value != BOMB):
            val = self.countBombs(row, col)
            self.board[row][col].value = val
            self.board[row][col].remainingValue = val

    def countBombs(self, r, c):
        """ Count the number of bombs around the cell """
        count = 0
        if(self.left(r, c) is not None and self.left(r, c).value == BOMB):
            count += 1
        if(self.right(r, c) is not None and self.right(r, c).value == BOMB):
            count += 1
        if(self.up(r, c) is not None and self.up(r, c).value == BOMB):
            count += 1
        if(self.down(r, c) is not None and self.down(r, c).value == BOMB):
            count += 1
        if(self.upLeft(r, c) is not None and self.upLeft(r, c).value == BOMB):
            count += 1
        if(self.upRight(r, c) is not None and self.upRight(r, c).value == BOMB):
            count += 1
        if(self.downLeft(r, c) is not None and self.downLeft(r, c).value == BOMB):
            count += 1
        if(self.downRight(r, c) is not None and self.downRight(r, c).value == BOMB):
            count += 1
        
        return count

    def left(self, row, col):
        if col == 0:
            return None
        else:
            return self.board[row][col-1]

    def right(self, row, col):
        if col == WIDTH-1:
            return None
        else:
            return self.board[row][col+1]

    def up(self, row, col):
        if row == 0:
            return None
        else:
            return self.board[row-1][col]

    def down(self, row, col):
        if row == HEIGHT-1:
            return None
        else:
            return self.board[row+1][col]

    def upLeft(self, row, col):
        if row == 0 or col == 0:
            return None
        else:
            return self.board[row-1][col-1]

    def upRight(self, row, col):
        if row == 0 or col == WIDTH-1:
            return None
        else:
            return self.board[row-1][col+1]

    def downLeft(self, row, col):
        if row == HEIGHT-1 or col == 0:
            return None
        else:
            return self.board[row+1][col-1]

    def downRight(self, row, col):
        if row == HEIGHT-1 or col == WIDTH-1:
            return None
        else:
            return self.board[row+1][col+1]

    def __str__(self):
        string = " "
        # Column headers
        for col in range(WIDTH):
            string += "{0:2}".format(col)
        string += "\n"
        
        # The actual rows
        row = 0
        for r in self.board:
            string += str(row)
            row += 1
            for tile in r:
                string += str(tile)
            string += "\n"
        
        return string
    
    def explore(self, row, col):
        """ Check the value of a cell """
        # Make the first move safe
        # if self.firstMove: # FIXME: I'm broken
        #     while True:
            # self.setupBoard()
        #         if self.board[row][col].value != BOMB:
        #             # print("It's safe to explore!", row, col) # TESTING
        #             break
            # self.firstMove = False

        if(not self.board[row][col].explored and not self.board[row][col].marked):
            value = self.board[row][col].explore()

            if(value == BOMB):
                print(self)
                print("YOU LOST :(")
                exit()

            if(value == 0):
                # Explore surrounding tiles
                if self.left(row, col) is not None:
                    if self.left(row, col).value == 0 and not self.left(row, col).marked:
                        self.explore(row, col-1)
                    elif self.left(row, col).value != BOMB and not self.left(row, col).marked:
                        self.left(row, col).explore()

                if self.right(row, col) is not None:
                    if self.right(row, col).value == 0 and not self.right(row, col).marked:
                        self.explore(row, col+1)
                    elif self.right(row, col).value != BOMB and not self.right(row, col).marked:
                        self.right(row, col).explore()

                if self.up(row, col) is not None:
                    if self.up(row, col).value == 0 and not self.up(row, col).marked:
                        self.explore(row-1, col)
                    elif self.up(row, col).value != BOMB and not self.up(row, col).marked:
                        self.up(row, col).explore()

                if self.down(row, col) is not None:
                    if self.down(row, col).value == 0 and not self.down(row, col).marked:
                        self.explore(row+1, col)
                    elif self.down(row, col).value != BOMB and not self.down(row, col).marked:
                        self.down(row, col).explore()
    
    def mark(self, row, col):
        """ Mark the tile as a bomb and update the number of correctly found bombs """
        self.board[row][col].mark()
        if self.board[row][col].marked and self.board[row][col].value == BOMB:
            self.correctMarks += 1

    def isSolved(self):
        """ Checks if the only remaining unexplored tiles are marked bombs """
        if self.correctMarks == MAX_BOMBS:
            marks = [tile.marked for row in self.board for tile in row]
            if marks.count(True) == MAX_BOMBS:
                return True
        return False