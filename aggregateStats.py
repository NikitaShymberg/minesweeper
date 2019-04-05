from game.constants import STATS_FILE, BOARDS, NUM_GAMES
import numpy as np
from sys import argv

if __name__ == "__main__":
    with open(STATS_FILE + argv[1], "r") as f:
        data = eval(f.read())

    moveNumbers = [[], [], [], [], []]
    wins = [[], [], [], [], []]
    explore = [[], [], [], [], []]
    moveTimes = [[], [], [], [], []]

    for d in data:
        if d["boardLayout"]["rows"] == BOARDS[0]["rows"]:
            moveNumbers[0].append(d["numMoves"])
            wins[0].append(int(d["win"]))
            explore[0].append(d["explored"])
            moveTimes[0] += d["moveTimes"]
        if d["boardLayout"]["rows"] == BOARDS[1]["rows"]:
            moveNumbers[1].append(d["numMoves"])
            wins[1].append(int(d["win"]))
            explore[1].append(d["explored"])
            moveTimes[1] += d["moveTimes"]
        if d["boardLayout"]["rows"] == BOARDS[2]["rows"]:
            moveNumbers[2].append(d["numMoves"])
            wins[2].append(int(d["win"]))
            explore[2].append(d["explored"])
            moveTimes[2] += d["moveTimes"]
        if d["boardLayout"]["rows"] == BOARDS[3]["rows"]:
            moveNumbers[3].append(d["numMoves"])
            wins[3].append(int(d["win"]))
            explore[3].append(d["explored"])
            moveTimes[3] += d["moveTimes"]
        if d["boardLayout"]["rows"] == BOARDS[4]["rows"]:
            moveNumbers[4].append(d["numMoves"])
            wins[4].append(int(d["win"]))
            explore[4].append(d["explored"])
            moveTimes[4] += d["moveTimes"]
    
    for i, arr in enumerate(moveNumbers):
        if arr == []:
            moveNumbers[i] = [0]
    for i, arr in enumerate(wins):
        if arr == []:
            wins[i] = [0]
    for i, arr in enumerate(explore):
        if arr == []:
            explore[i] = [0]
    for i, arr in enumerate(moveTimes):
        if arr == []:
            moveTimes[i] = [0]

    moveNumbers = np.array(moveNumbers)
    wins = np.array(wins)
    explore = np.array(explore)
    moveTimes = np.array(moveTimes)

    print("            Average # moves in xs:", "%3.3f" % np.mean(moveNumbers[0]), "Median:", "%3.3f" % np.median(moveNumbers[0]), "Std:", "%3.3f" % np.std(moveNumbers[0]), "Min:", "%3.3f" % np.min(moveNumbers[0]), "Max:", "%3.3f" % np.max(moveNumbers[0]))
    print("            Average # moves in s: ", "%3.3f" % np.mean(moveNumbers[1]), "Median:", "%3.3f" % np.median(moveNumbers[1]), "Std:", "%3.3f" % np.std(moveNumbers[1]), "Min:", "%3.3f" % np.min(moveNumbers[1]), "Max:", "%3.3f" % np.max(moveNumbers[1]))
    print("            Average # moves in m: ", "%3.3f" % np.mean(moveNumbers[2]), "Median:", "%3.3f" % np.median(moveNumbers[2]), "Std:", "%3.3f" % np.std(moveNumbers[2]), "Min:", "%3.3f" % np.min(moveNumbers[2]), "Max:", "%3.3f" % np.max(moveNumbers[2]))
    print("            Average # moves in l: ", "%3.3f" % np.mean(moveNumbers[3]), "Median:", "%3.3f" % np.median(moveNumbers[3]), "Std:", "%3.3f" % np.std(moveNumbers[3]), "Min:", "%3.3f" % np.min(moveNumbers[3]), "Max:", "%3.3f" % np.max(moveNumbers[3]))
    print("            Average # moves in xl:", "%3.3f" % np.mean(moveNumbers[4]), "Median:", "%3.3f" % np.median(moveNumbers[4]), "Std:", "%3.3f" % np.std(moveNumbers[4]), "Min:", "%3.3f" % np.min(moveNumbers[4]), "Max:", "%3.3f" % np.max(moveNumbers[4]))
    print()
    print("            Percentage wins in xs:", "%3.3f" % np.mean(wins[0]), "Median:", "%3.3f" % np.median(wins[0]), "Std:", "%3.3f" % np.std(wins[0]), "Min:", "%3.3f" % np.min(wins[0]), "Max:", "%3.3f" % np.max(wins[0]))
    print("            Percentage wins in s: ", "%3.3f" % np.mean(wins[1]), "Median:", "%3.3f" % np.median(wins[1]), "Std:", "%3.3f" % np.std(wins[1]), "Min:", "%3.3f" % np.min(wins[1]), "Max:", "%3.3f" % np.max(wins[1]))
    print("            Percentage wins in m: ", "%3.3f" % np.mean(wins[2]), "Median:", "%3.3f" % np.median(wins[2]), "Std:", "%3.3f" % np.std(wins[2]), "Min:", "%3.3f" % np.min(wins[2]), "Max:", "%3.3f" % np.max(wins[2]))
    print("            Percentage wins in l: ", "%3.3f" % np.mean(wins[3]), "Median:", "%3.3f" % np.median(wins[3]), "Std:", "%3.3f" % np.std(wins[3]), "Min:", "%3.3f" % np.min(wins[3]), "Max:", "%3.3f" % np.max(wins[3]))
    print("            Percentage wins in xl:", "%3.3f" % np.mean(wins[4]), "Median:", "%3.3f" % np.median(wins[4]), "Std:", "%3.3f" % np.std(wins[4]), "Min:", "%3.3f" % np.min(wins[4]), "Max:", "%3.3f" % np.max(wins[4]))
    print()
    print("Average percentage explored in xs:", "%3.3f" % np.mean(explore[0]), "Median:", "%3.3f" % np.median(explore[0]), "Std:", "%3.3f" % np.std(explore[0]), "Min:", "%3.3f" % np.min(explore[0]), "Max:", "%3.3f" % np.max(explore[0]))
    print("Average percentage explored in s: ", "%3.3f" % np.mean(explore[1]), "Median:", "%3.3f" % np.median(explore[1]), "Std:", "%3.3f" % np.std(explore[1]), "Min:", "%3.3f" % np.min(explore[1]), "Max:", "%3.3f" % np.max(explore[1]))
    print("Average percentage explored in m: ", "%3.3f" % np.mean(explore[2]), "Median:", "%3.3f" % np.median(explore[2]), "Std:", "%3.3f" % np.std(explore[2]), "Min:", "%3.3f" % np.min(explore[2]), "Max:", "%3.3f" % np.max(explore[2]))
    print("Average percentage explored in l: ", "%3.3f" % np.mean(explore[3]), "Median:", "%3.3f" % np.median(explore[3]), "Std:", "%3.3f" % np.std(explore[3]), "Min:", "%3.3f" % np.min(explore[3]), "Max:", "%3.3f" % np.max(explore[3]))
    print("Average percentage explored in xl:", "%3.3f" % np.mean(explore[4]), "Median:", "%3.3f" % np.median(explore[4]), "Std:", "%3.3f" % np.std(explore[4]), "Min:", "%3.3f" % np.min(explore[4]), "Max:", "%3.3f" % np.max(explore[4]))
    print()
    print("         Average move times in xs:", "%3.3f" % np.mean(moveTimes[0]), "Median:", "%3.3f" % np.median(moveTimes[0]), "Std:", "%3.3f" % np.std(moveTimes[0]), "Min:", "%3.3f" % np.min(moveTimes[0]), "Max:", "%3.3f" % np.max(moveTimes[0]))
    print("         Average move times in s: ", "%3.3f" % np.mean(moveTimes[1]), "Median:", "%3.3f" % np.median(moveTimes[1]), "Std:", "%3.3f" % np.std(moveTimes[1]), "Min:", "%3.3f" % np.min(moveTimes[1]), "Max:", "%3.3f" % np.max(moveTimes[1]))
    print("         Average move times in m: ", "%3.3f" % np.mean(moveTimes[2]), "Median:", "%3.3f" % np.median(moveTimes[2]), "Std:", "%3.3f" % np.std(moveTimes[2]), "Min:", "%3.3f" % np.min(moveTimes[2]), "Max:", "%3.3f" % np.max(moveTimes[2]))
    print("         Average move times in l: ", "%3.3f" % np.mean(moveTimes[3]), "Median:", "%3.3f" % np.median(moveTimes[3]), "Std:", "%3.3f" % np.std(moveTimes[3]), "Min:", "%3.3f" % np.min(moveTimes[3]), "Max:", "%3.3f" % np.max(moveTimes[3]))
    print("         Average move times in xl:", "%3.3f" % np.mean(moveTimes[4]), "Median:", "%3.3f" % np.median(moveTimes[4]), "Std:", "%3.3f" % np.std(moveTimes[4]), "Min:", "%3.3f" % np.min(moveTimes[4]), "Max:", "%3.3f" % np.max(moveTimes[4]))
    print()
    # TODO: max, min, stdev