
import math
from dataclasses import dataclass, field
from random import Random
from random import randrange

#for adjusting the move limit if the formual used gives a too-high limit
decrement_percentage : float = 37

@dataclass
class Config:
    #default size and colors
    size: int = 3
    colors: int = 4

    #for max move limit
    left: float = field(init=False, default=0.0)
    right: float = field(init=False, default=0.0)
    move_limit: int = field(init=False, default=0)

    # after the size/colors fields are set, derive the move limit
    def __post_init__(self):
        """
            the formula used here is based on the research paper I had read
        """

        # lower bound of the recommended move interval
        self.left = (math.sqrt(self.colors - 1) * self.size / 2) - self.colors / 2
        # upper bound of the recommended move interval
        self.right = 2 * self.size + (math.sqrt(2 * self.colors) * self.size) + self.colors
        #taking a floor of the avg to get the move limit
        self.avg_ = int((self.left + self.right) / 2)

        # using the decrement percentage to adjust the move limit
        self.move_limit = int(self.avg_ - self.avg_ * (decrement_percentage/100)) - 2


# Board configuration and state
class Board:
    def __init__(self, config: Config, grid=None, moves_used: int = 0, rng=None):
        #for board state and randomness
        self.config = config
        self.rng = rng
        self.grid = grid if grid is not None else self._random_grid()
        # count of valid flood moves the player has made so far
        self.moves_used = moves_used

    # builds a fresh random puzzle grid
    def _random_grid(self):
        n, c = self.config.size, self.config.colors
        if self.rng is not None:
            #generates random colors for each cell in the grid
            return [[self.rng.randrange(c) for _ in range(n)] for _ in range(n)]

        # otherwise fall back to the module-level random (unseeded behaviour)
        return [[randrange(c) for _ in range(n)] for _ in range(n)]

    # finds every cell connected to the top-left corner with the same color
    def connected(self):
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

                    # add neighbour to the frontier if it matches the start color
                    if self.grid[nr][nc] == start_color:
                        # mark that neighbour as seen
                        seen.add((nr, nc))

                        # now for checking if that neighbout cell has more neighbours to explore for color match
                        frontier.append((nr, nc))

        # returns the full list of connected same-color cells
        return seen

    # flooding sequence
    def flood(self, color: int) -> bool:

        # reject invalid color numbers
        if not (0 <= color < self.config.colors):
            return False

        # can't flood the same color as the top-left cell
        if color == self.grid[0][0]:
            return False

        # recolor every cell in the current connected region
        for r, c in self.connected():
            self.grid[r][c] = color

        self.moves_used += 1
        return True

    # checking for the win condition
    def is_solved(self):

        first = self.grid[0][0]
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

    # to check how many cells the flood currently owns
    @property
    def coverage(self):
        # every cell in the connected region is owned by the flood
        return len(self.connected())


    # independent copy of the for board state
    def clone(self):

        return Board(self.config,
                     grid=[row[:] for row in self.grid],
                     moves_used=self.moves_used,
                     rng=self.rng)
