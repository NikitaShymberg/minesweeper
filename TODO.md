# TODO:
1. Imports are wack
2. Rename vars to _ instead of camel?
3. Document functions
4. Numpy might make things faster
   1. Get rid of as many appends as you can
5. Refactor some things
6. LINT!!
7. Keep checking what kinds of mistakes it makes

Third algorithm: https://github.com/jakejhansen/minesweeper_solver

TESTING CRITERIA:
1. Win rate
   1. Can ignore games where you lose on the second move perhaps
2. Stats on time taken to make a move
   1. Log the time whever a move happens, then process stuff afterwards
3. Percentage of board explored
4. Scalability
   1. 4x4 3 mines
   2. 8x8 10 mines
   3. 16x16 40 mines
   4. 24x24 99 mines
   5. 50x50 400 mines (bonus?)
5. Number of moves
   1. More is better
   2. If you win then fewer is better but that's an edge case
6. Memory usage
   1. This will be different for things on a gpu
7. Cpu/gpu(?) usage
8. Maybe how well it runs on cpu
9. A comment on "smartness"
   1.  Maybe whether the mistakes it makes are acceptable but idk
10. Compare to a random move each time
    1.  Mention how comparing to human performance bfs is just way better

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