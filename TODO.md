# TODO:
1. Imports are wack
2. Rename vars to _ instead of camel?
3. Document functions
4. Numpy might make things faster
   1. Get rid of as many appends as you can
5. Refactor some things
6. LINT!!
7. Train it on custom data
   2. Certain bomb setups for 1, 2, 3, 4, ... Overfit these hella
        E E E + +
        E 1 E + +
        E E * + +
        + + + + +
        + + + + +
        
        + E E E +
        + E 1 E +
        + E * E +
        + + + + +
        + + + + +
        
        E E E + +
        E 2 E + +
        E * * + +
        + + + + +
        + + + + +
        
        E E E + +
        E 3 * + +
        E * * + +
        + + + + +
        + + + + +
        
        E E * + +
        E 4 * + +
        E * * + +
        + + + + +
        + + + + +
        
   1. Fully/mostly explored? That's where it dies a lot?

Generating data:
    Make a board with a bunch of bombs
    Explore some tiles randomly
    getTilesAdjacentToExploredTiles()
    process them
    return that
    do this on demand

Interesting Examples:

  0 1 2 3 4 5 6 7
0 0 1 . . . . . .
1 0 2 . . . . . .
2 0 2 . . . . . .
3 0 1 2 3 2 1 . .
4 0 0 0 0 0 0 1 .
5 1 1 1 0 0 0 1 .
6 . . . 1 2 2 . .
7 . . . . . . . .

Number of bombs left: 10
Number of tiles to consider: 21
Total unexplored tiles: 35
3 6 0
3 7 -1
4 7 0
5 7 0
6 0 0
1 2 -1
6 2 0
2 4 -1
6 6 0
6 7 -1
7 5 -1
2 3 -1
7 4 0
2 5 0
7 2 0
2 6 0
7 3 -1
6 1 -1
7 6 -1
0 2 0
2 2 -1
--------------------------------
3 6 0
3 7 0
4 7 -1
5 7 0
6 0 0
1 2 -1
6 2 0
2 4 -1
6 6 0
6 7 0
7 5 -1
2 3 -1
7 4 -1
2 5 0
7 2 0
2 6 0
7 3 0
6 1 -1
7 6 0
0 2 0
2 2 -1
--------------------------------
3 6 0
3 7 0
4 7 0
5 7 -1
6 0 0
1 2 -1
6 2 0
2 4 -1
6 6 0
6 7 0
7 5 -1
2 3 -1
7 4 -1
2 5 0
7 2 0
2 6 0
7 3 0
6 1 -1
7 6 0
0 2 0
2 2 -1
--------------------------------
3 6 0
3 7 -1
4 7 0
5 7 0
6 0 0
1 2 -1
6 2 0
2 4 -1
6 6 0
6 7 -1
7 5 -1
2 3 -1
7 4 -1
2 5 0
7 2 0
2 6 0
7 3 0
6 1 -1
7 6 0
0 2 0
2 2 -1
--------------------------------
3 6 0
3 7 0
4 7 -1
5 7 0
6 0 0
1 2 -1
6 2 0
2 4 -1
6 6 0
6 7 0
7 5 -1
2 3 -1
7 4 0
2 5 0
7 2 0
2 6 0
7 3 -1
6 1 -1
7 6 -1
0 2 0
2 2 -1
--------------------------------
3 6 0
3 7 0
4 7 0
5 7 -1
6 0 0
1 2 -1
6 2 0
2 4 -1
6 6 0
6 7 0
7 5 -1
2 3 -1
7 4 0
2 5 0
7 2 0
2 6 0
7 3 -1
6 1 -1
7 6 -1
0 2 0
2 2 -1
--------------------------------
Number of times this tile was a bomb: [0, 2, 2, 2, 0, 6, 0, 6, 0, 2, 6, 6, 3, 0, 0, 0, 3, 6, 3, 0, 6]
Probability of this tile being a bomb: [0.0, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.0, 1.0, 0.0, 1.0, 0.0, 0.3333333333333333,1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 1.0]
Number of bombs in each permutation: [10, 8, 8, 9, 9, 9]




  0 1 2 3 4 5 6 7
0 0 0 0 1 . . . .
1 1 2 2 . . . . .
2 . . . . . . . .
3 . . . . . . . .
4 . . . . . . . .
5 . . . . . . . .
6 . . . . . . . .
7 . . . . . . . .

Number of bombs left: 10
Number of tiles to consider: 7
Total unexplored tiles: 57
1 3 0
0 4 -1
2 2 -1
1 4 0
2 3 -1
2 0 -1
2 1 0
--------------------------------
1 3 0
0 4 0
2 2 -1
1 4 -1
2 3 -1
2 0 -1
2 1 0
--------------------------------
1 3 0
0 4 -1
2 2 -1
1 4 0
2 3 0
2 0 0
2 1 -1
--------------------------------
1 3 0
0 4 0
2 2 -1
1 4 -1
2 3 0
2 0 0
2 1 -1
--------------------------------
Number of times this tile was a bomb: [0, 2, 4, 2, 2, 2, 2]
Probability of this tile being a bomb: [0.0, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5]
Number of bombs in each permutation: [4, 4, 3, 3]