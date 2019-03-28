BOMB = -1
WIDTH = 8
HEIGHT = 8
MAX_BOMBS = 10

BATCH_SIZE = 2000 # CHANGED TO 4000 ON CC
EPOCHS = 1000000
LR = 5e-6
# REG = 1e-10
REG = 2e-6
CHECKPOINT_FILE = "runs/checkpoint.pth"
MODEL = "2dnnNEW"
MOVE_CERTAINTY_THRESHOLD = 0 # TODO: determine me!
MARK_CERTAINTY_THRESHOLD = 0 # TODO: determine me!
EXPLORE_COEFF = 1 # TODO: determine me!

NUM_GAMES = [100, 1000, 1000, 100, 100]
BOARDS = [
    {"height": 4, "width": 4, "numBombs": 3},
    {"height": 8, "width": 8, "numBombs": 10},
    {"height": 16, "width": 16, "numBombs": 40},
    {"height": 24, "width": 24, "numBombs": 99},
    {"height": 50, "width": 50, "numBombs": 400},
]