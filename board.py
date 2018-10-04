from constants import WIDTH, HEIGHT, BOMB
from random import randint


class Board:
    board = []

    def __init__(self):
        self.board = [[0 for i in range(WIDTH)] for j in range(HEIGHT)]
        self.setupBoard()

    def setupBoard(self):
        setBombs = 0
        # NOTE: might be slow if bomb dense
        while(setBombs < 10):
            row = randint(0, WIDTH-1)
            col = randint(0, HEIGHT-1)
            if(self.board[row][col] != BOMB):
                self.board[row][col] = BOMB
                setBombs += 1
        
        # TODO: improve
        for r in range(WIDTH):
            for c in range(HEIGHT):
                self.updateCounts(r, c)

    def updateCounts(self, row, col):
        """ Given a cell, update its value """
        if(self.board[row][col] != BOMB):
            self.board[row][col] = self.countBombs(row, col)

    def countBombs(self, r, c):
        """ Count the number of bombs around the cell """
        count = 0
        if(self.left(r, c) == BOMB):
            count += 1
        if(self.right(r, c) == BOMB):
            count += 1
        if(self.up(r, c) == BOMB):
            count += 1
        if(self.down(r, c) == BOMB):
            count += 1
        if(self.upLeft(r, c) == BOMB):
            count += 1
        if(self.upRight(r, c) == BOMB):
            count += 1
        if(self.downLeft(r, c) == BOMB):
            count += 1
        if(self.downRight(r, c) == BOMB):
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
        if row == HEIGHT-1:
            return None
        else:
            return self.board[row+1][col]

    def down(self, row, col):
        if row == 0:
            return None
        else:
            return self.board[row-1][col]

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
    
    def print(self):
        for r in self.board:
            s = ""
            for c in r:
                s += "%2s"%c
            print(s)