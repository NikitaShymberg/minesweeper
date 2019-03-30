BOMB = -1
# WIDTH = 8
# HEIGHT = 8
# MAX_BOMBS = 10

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

# NUM_GAMES = [10]
# BOARDS = [
#     {"rows": 4, "cols": 4, "numBombs": 3},
# ]
NUM_GAMES = [1000, 50, 10, 10, 10]
BOARDS = [
    {"rows": 4, "cols": 4, "numBombs": 3},
    {"rows": 8, "cols": 8, "numBombs": 10},
    {"rows": 16, "cols": 16, "numBombs": 40},
    {"rows": 24, "cols": 24, "numBombs": 99},
    {"rows": 50, "cols": 50, "numBombs": 400},
]

STATS_FILE = "stats_"