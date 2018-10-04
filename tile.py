from constants import *

class Tile:
    def __init__(self, value):
        self.explored = False
        self.marked = False
        self.value = value
    
    def mark(self):
        self.marked = not self.marked
    
    def explore(self):
        self.explored = True
        return self.value
    
    def print(self):
        if self.explored:
            if self.value == BOMB:
                print(" *", end='')
            else:
                print("%2s"%self.value, end='')
        else:
            print(" .", end='')
