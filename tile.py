from constants import *

class Tile:
    def __init__(self, value):
        self.explored = False
        self.marked = False
        self.value = value
    
    def mark():
        self.marked = not self.marked
    
    def explore():
        return self.value