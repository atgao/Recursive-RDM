# Recursive-RDM
Code for senior thesis evaluating recursive tournament rules

## Requirements
- Python 3.5+
- numpy

## Running the code
- run `python RDM.py -n [n] -s [s] -t [True/False]` in the main directory
- `n` is the starting number of nodes for the graph (default `n=4`)
- `s` is the number of colluders (defaul `s=3`)
- to run RDM on only graphs of size `n`, give the False argument for `t`
- the current algorithm will automatically increment until `n=10`


## Notes 
There were a few minor issues with getting the `nauty` package (to use `gentourng`) to work on WSL/Ubuntu. Make sure that everything inside the `\nauty27r1` folder has executable permissions.