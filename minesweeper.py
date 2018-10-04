from board import Board
from constants import WIDTH, HEIGHT, MAX_BOMBS
# -1 = BOMB
# number = number of bombs near me

board = Board()

flags = MAX_BOMBS
while(flags > 0):
    board.print()
    row = int(input("ROW:"))
    col = int(input("COL:"))
    board.explore(row, col)
