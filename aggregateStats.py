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
    
    print("Average # moves in xs:", sum(moveNumbers[0]) / NUM_GAMES[0])
    print("Average # moves in s:", sum(moveNumbers[1]) / NUM_GAMES[1])
    print("Average # moves in m:", sum(moveNumbers[2]) / NUM_GAMES[2])
    print("Average # moves in l:", sum(moveNumbers[3]) / NUM_GAMES[3])
    print("Average # moves in xl:", sum(moveNumbers[4]) / NUM_GAMES[4])
    print()
    print("Percentage wins in xs:", sum(wins[0]) / NUM_GAMES[0])
    print("Percentage wins in s:", sum(wins[1]) / NUM_GAMES[1])
    print("Percentage wins in m:", sum(wins[2]) / NUM_GAMES[2])
    print("Percentage wins in l:", sum(wins[3]) / NUM_GAMES[3])
    print("Percentage wins in xl:", sum(wins[4]) / NUM_GAMES[4])
    print()
    print("Average percentage explored in xs:", sum(explore[0]) / NUM_GAMES[0])
    print("Average percentage explored in s:", sum(explore[1]) / NUM_GAMES[1])
    print("Average percentage explored in m:", sum(explore[2]) / NUM_GAMES[2])
    print("Average percentage explored in l:", sum(explore[3]) / NUM_GAMES[3])
    print("Average percentage explored in xl:", sum(explore[4]) / NUM_GAMES[4])
    print()
    print("Average move times in xs:", sum(moveTimes[0]) / NUM_GAMES[0])
    print("Average move times in s:", sum(moveTimes[1]) / NUM_GAMES[1])
    print("Average move times in m:", sum(moveTimes[2]) / NUM_GAMES[2])
    print("Average move times in l:", sum(moveTimes[3]) / NUM_GAMES[3])
    print("Average move times in xl:", sum(moveTimes[4]) / NUM_GAMES[4])
    print()
    # TODO: max, min, stdev