# Flood-It RL environment: bridges the game logic to any learning algorithm.
#
# The API mimics OpenAI Gym / Gymnasium (reset / step / render / close) so that
# the same environment can later be plugged into gym, Stable-Baselines3, etc.
# without rewriting it. It uses only the standard library (no numpy/gymnasium).

# 'Random' builds a seedable RNG so episodes can be reproduced exactly
from random import Random

# import the game logic classes from the sibling flood_it module
from flood_it import Board, Config


# Minimal stand-in for gym.spaces.Discrete: an action is one of 0..n-1.
class Discrete:
    # constructor: store how many distinct choices the space allows
    def __init__(self, n: int):
        self.n = n

    # pick a random valid action using a supplied RNG (avoids global state)
    def sample(self, rng: Random) -> int:
        return rng.randrange(self.n)

    # membership test so the env can validate a received action
    def __contains__(self, action):
        return isinstance(action, int) and 0 <= action < self.n

    # human-readable description of the space
    def __repr__(self):
        return f"Discrete({self.n})"


# Minimal stand-in for gym.spaces.Box: describes a range of integer values.
# Used here to describe the board grid portion of the observation.
class Box:
    # constructor: 'shape' is the grid dimensions, low/high the value bounds
    def __init__(self, shape, low: int, high: int):
        self.shape = shape
        self.low = low
        self.high = high

    # human-readable description of the space
    def __repr__(self):
        return f"Box({self.shape}, {self.low}, {self.high})"


# Minimal stand-in for gym.spaces.Tuple: a fixed-size list of sub-spaces.
# The observation is (grid, moves_left), so we combine a Box and a scalar Box.
class Tuple:
    # constructor: store the ordered list of component spaces
    def __init__(self, *spaces):
        self.spaces = spaces

    # human-readable description of the space
    def __repr__(self):
        return f"Tuple({', '.join(repr(s) for s in self.spaces)})"


# The reinforcement-learning environment around the Flood-It game.
class FloodItEnv:
    # constructor: build the configuration and the action/observation spaces
    #   config:       optional ready-made Config (overridden by any kwarg given)
    #   size/colors/move_limit: keyword overrides, applied on top of a default
    #   win_reward:   bonus added to the shaped reward when the board is solved
    #   loss_reward:  penalty added when the agent runs out of moves unsolved
    def __init__(self, config: Config | None = None, size=None, colors=None,
                 move_limit=None, win_reward: float = 1.0, loss_reward: float = -1.0):
        # start from the caller's Config or a default-sized one
        cfg = config if config is not None else Config()
        # apply any keyword overrides on top of that base configuration
        if size is not None:
            cfg.size = size
        if colors is not None:
            cfg.colors = colors
        if move_limit is not None:
            cfg.move_limit = move_limit
        # keep the final settings for constructing every new board
        self.config = cfg

        # reward tuning knobs stored for use inside step()
        self.win_reward = win_reward
        self.loss_reward = loss_reward

        # no board exists until reset() is called -> the env starts "unplayed"
        self.board = None
        # RNG used to generate fresh boards; created from a seed inside reset()
        self.rng = None
        # flag set True once the game has ended (win or out of moves)
        self.done = True

        # describe the action space: pick any one of the available colors
        self.action_space = Discrete(cfg.colors)
        # describe the observation space: a (size x size) grid of colors 0..c-1,
        # followed by a scalar move budget
        self.observation_space = Tuple(
            Box((cfg.size, cfg.size), 0, cfg.colors - 1),
            Box((), 0, cfg.move_limit),
        )

    # start a fresh episode: build a new seeded board, reset the done flag
    def reset(self, seed=None):
        # create a fresh RNG (same seed -> same board every time)
        self.rng = Random(seed)
        # build a brand-new random board using that seeded RNG
        self.board = Board(self.config, rng=self.rng)
        # the game is now live again
        self.done = False
        # return the initial observation plus an info dict for the caller
        return self.board.observe(), self._info()

    # apply one action and return (obs, reward, done, truncated, info)
    def step(self, action: int):
        # enforce the gym rule: the environment must be reset before stepping
        if self.board is None:
            raise RuntimeError("call reset() before step()")
        # if the game already ended, return a zero-reward "do nothing" signal
        if self.done:
            return self.board.observe(), 0.0, True, True, self._info()
        # reject actions that fall outside the available color range
        if action not in self.action_space:
            raise ValueError(f"action {action} out of range 0..{self.action_space.n - 1}")

        # record how much of the board the flood owned before the move
        before = self.board.coverage
        # the color currently flooding the corner region
        region = self.board.grid[0][0]

        # no-op move: re-picking the region color changes nothing
        if action == region:
            # it is allowed but it still burns a turn (costs a move)
            self.board.moves_used += 1
            # nothing was gained, so the shaped reward is zero
            gained = 0
        else:
            # real move: flood the region with the chosen color
            self.board.flood(action)
            # shaped reward = how many new cells the flood captured this move
            gained = self.board.coverage - before

        # shaped reward: the more cells captured, the better the move
        reward = float(gained)
        # win bonus: reward spikes when the board becomes a single color
        if self.board.is_solved():
            reward += self.win_reward
        # loss penalty: no moves left and the board is not yet solved
        elif self.board.moves_left == 0:
            reward += self.loss_reward

        # update the done flag: solved, or ran out of moves
        self.done = self.board.is_over()

        # return the full gym-style result tuple for the agent
        return self.board.observe(), reward, self.done, False, self._info()

    # which colors actually change the board (everything except the region color)
    @property
    def legal_actions(self):
        # the color flooding the corner right now
        region = self.board.grid[0][0]
        # every other color is a move that would actually flood new cells
        return [c for c in range(self.config.colors) if c != region]

    # build the extra per-step info dict passed to the agent
    def _info(self):
        # 1 means "this color would change the board", 0 means no-op
        action_mask = [0 if c == self.board.grid[0][0] else 1
                       for c in range(self.config.colors)]
        # bundle all useful bookkeeping together for the agent
        return {
            "moves_left": self.board.moves_left,
            "moves_used": self.board.moves_used,
            "flooded_cells": self.board.coverage,
            "solved": self.board.is_solved(),
            "action_mask": action_mask,
        }

    # draw the board as plain text (headless-safe; no tkinter involved)
    def render(self):
        # print one row per line, cells joined by spaces for readability
        for row in self.board.grid:
            print(" ".join(str(cell) for cell in row))

    # gym compatibility hook; nothing to release since we hold no resources
    def close(self):
        pass
