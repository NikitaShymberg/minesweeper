WIDTH = 8
HEIGHT = 8
MAX_BOMBS = 10
BOMB = -1

BATCH_SIZE = 2000
EPOCHS = 1000000
LR = 5e-5
# REG = 1e-10
REG = 2e-6
CHECKPOINT_FILE = "runs/checkpoint.pth"
BEST_FILE = "runs/best.pth"
MODEL = "2dnn"
MOVE_CERTAINTY_THRESHOLD = 3 # TODO: determine me!
MARK_CERTAINTY_THRESHOLD = 8 # TODO: determine me!