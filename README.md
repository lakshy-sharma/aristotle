# Aristotle

The arsitotle is a project where I conduct practical research.

The code has been written such that each research project can be integrated into the mainline code base using plugin registration.
Only requirement is that the programmers follow a basic naming convention.

The program layout is simple with the following folder.

1. modules
2. views
3. inputs
4. outputs: When in offline mode all the 
5. server: Once a module is ready to be published you can quickly create a fastapi based server to serve your modules.

## Current Initiatives

Currently I am working on two research problems.

1. Prisoners Dilemma Based Game Theory research.
2. Sokoban solver.

### Prisoners Dilemma

This is a simulator + analyser for the classic game of prisoners dilemma in game theory.

#### Setup
To start playing the game you must perform the following steps.

```
sudo apt install ffmpeg virtualenv
virtualenv isolation
source isolation/bin/activate
pip3 install -r requirements.txt
```

#### Registered Players
1. Jesus: A player who never retaliates
2. Judas: An envious player who only cooperates while it is winning. Otherwise it defects.
3. Tit for Tat: A player who starts out by cooperating but retaliates immediately when deceived.
4. Joker: A random player who believes in anarchy and plays with a random strategy. Because Chaos is Fair...
5. Lucifer: A player who always defects.
6. Worse and Worse: A player who defects more as the game proceeds. Essentially drawing out the worst in its opponents as the game proceeds.
7. Better and Better: A player who cooperates mre as the game goes on. Essentially bringing out the bestv in their opponents as the game proceeds.
8. Grudger: A player who holds grudges and stops cooperating once an opponent defects.

### Sokoban Solver

This is a solevr for sokoban puzzles.
It is an initiative to further the research in artificial intelligence to plan future moves with heuristics without bruteforcing.
This module is under progress.

## Future Plans

1. Create an Nbody simulator.