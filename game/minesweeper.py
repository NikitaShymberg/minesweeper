from board import Board
from constants import WIDTH, HEIGHT, MAX_BOMBS
# -1 = BOMB
# number = number of bombs near me

board = Board()

won = False
while(not won):
    print(board)
    mode = int(input("Type 1 to explore or 0 to mark a bomb: "))
    if mode == 1:
        print("Explore: ")
        row = int(input("ROW:"))
        col = int(input("COL:"))
        board.explore(row, col)
    elif mode == 0:
        print("Mark a bomb: ")
        row = int(input("ROW:"))
        col = int(input("COL:"))
        board.mark(row, col)
    else:
        print("Error: Invalid mode")
    won = board.isSolved()

print("Congratulations, you won!")