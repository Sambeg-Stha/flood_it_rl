

# 'dataclass' auto-generates __init__/repr/eq for the Config class below
from dataclasses import dataclass
# 'randrange' picks a random integer from a range -> used to build random boards
from random import randrange


# @dataclass decorator turns Config into a small data container
@dataclass
class Config:
    # number of cells per side of the square grid (14x14 by default)
    size: int = 14
    # how many distinct colors the board can contain (default 6)
    colors: int = 6
    # how many moves the player is allowed before losing (default 25)
    move_limit: int = 25


# Board holds all game state and the rules for playing
class Board:
    # constructor: takes a Config plus an optional pre-made grid
    def __init__(self, config: Config, grid=None):
        # store the configuration for later use by every other method
        self.config = config
        # use the injected grid if given, otherwise create a random one
        self.grid = grid if grid is not None else self._random_grid()
        # count of valid flood moves the player has made so far
        self.moves_used = 0

    # builds a fresh random puzzle grid
    def _random_grid(self):
        # unpack the grid size and color count into short local names
        n, c = self.config.size, self.config.colors
        # create n rows, each with n cells, every cell a random color 0..c-1
        return [[randrange(c) for _ in range(n)] for _ in range(n)]

    # finds every cell connected to the top-left corner with the same color
    def connected(self):
        # local alias for grid size so it is quick to reference below
        n = self.config.size
        # color of the corner cell, which defines the current flooded region
        start_color = self.grid[0][0]
        # set of cells already known to be part of the region (starts at corner)
        seen = {(0, 0)}
        # stack of cells whose neighbors still need to be checked
        frontier = [(0, 0)]
        # keep expanding until there are no more cells to explore
        while frontier:
            # pop one cell off the stack (last-in-first-out => depth-first search)
            r, c = frontier.pop()
            # loop over the four neighbors: up, down, left, right
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                # compute the neighbor's row and column
                nr, nc = r + dr, c + dc
                # only consider neighbors inside the grid that we haven't visited
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in seen:
                    # the neighbor is only part of the region if its color matches
                    if self.grid[nr][nc] == start_color:
                        # mark it as part of the region
                        seen.add((nr, nc))
                        # and queue it up so its own neighbors get checked too
                        frontier.append((nr, nc))
        # hand back the full set of connected same-color cells
        return seen

    # makes a flood move: recolor the connected region to 'color'
    def flood(self, color: int) -> bool:
        # reject out-of-range color numbers (invalid move)
        if not (0 <= color < self.config.colors):
            return False
        # rejecting the current color makes no move and would waste a turn
        if color == self.grid[0][0]:
            return False
        # recolor every cell in the current connected region
        for r, c in self.connected():
            self.grid[r][c] = color
        # the move was valid, so count it
        self.moves_used += 1
        # report that the move was applied
        return True

    # True when every cell on the board shares one color (win condition)
    def is_solved(self):
        # the target color is the top-left cell's color
        first = self.grid[0][0]
        # solved only if each cell across every row equals that color
        return all(cell == first for row in self.grid for cell in row)

    # read-only property: how many moves the player still has
    @property
    def moves_left(self):
        # remaining = limit minus used, clamped so it never goes negative
        return max(0, self.config.move_limit - self.moves_used)

    # True once the game has ended, whether by winning or running out of moves
    def is_over(self):
        # game is over if solved, or if no moves remain
        return self.is_solved() or self.moves_left <= 0
