from constants import BOMB

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
    
    def __str__(self):
        # TODO: dynamic widths
        if self.explored:
            if self.value == BOMB:
                return " *"
            else:
                return "{0:2}".format(self.value)
        elif self.marked:
            return " X"
        else:
            return " ."
