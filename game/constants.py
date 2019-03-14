BOMB = -1
WIDTH = 8
HEIGHT = 8
MAX_BOMBS = 10

BATCH_SIZE = 2000
EPOCHS = 1000000
LR = 5e-5
# REG = 1e-10
REG = 2e-6
CHECKPOINT_FILE = "runs/checkpoint.pth"
MODEL = "2dnnNEW"
MOVE_CERTAINTY_THRESHOLD = 0.8 # TODO: determine me!
MARK_CERTAINTY_THRESHOLD = 0.9 # TODO: determine me!
EXPLORE_COEFF = 1 # TODO: determine me!
