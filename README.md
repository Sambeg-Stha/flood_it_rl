# Flood-It Using Max Covrage Algoritm

## Files

| File | Purpose |
|------|---------|
| `flood_it.py` | Core game logic: `Config` and `Board` |
| `main.py` | `tkinter` GUI for playing the game interactively. |
| `solver.py` | `GreedyAgent` that choses moves based on max coverage|


Run `python main.py` to play the game interactively.
Optional flags:

# The formula used fo defining the maximun move limit is based on the Research Paper "The Complexity of Flood Filling Games" 
The move limit is derived automatically from the size and colors using an interval formula (lower bound `sqrt(colors-1) * size/2 - colors/2`, upper bound `2*size + sqrt(2*colors)*size + colors`), then the midpoint is reduced by a tuning percentage (default 37%) defined in `flood_it.py`.

## How the game works

- The flood region is the set of cells connected to the top-left corner that share its color, found by a depth-first search (`Board.connected`).
- `Board.flood(color)` recolors that region and increments the move counter.
- The game is won when every cell matches the top-left color (`Board.is_solved`), and lost when the move limit is exhausted.

## How the solver works
`GreedyAgent.select_move(board)` clones the board, simulates flooding with every color except the one already owned, and returns the color that yields the largest flooded region.
