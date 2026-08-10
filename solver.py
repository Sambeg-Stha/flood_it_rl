# Greedy max-coverage solver for Flood-It

# argparse handles the command-line flags like --size and --colors
import argparse

# 'Random' lets us build a seedable RNG so every episode is reproducible
from random import Random

# import the game logic classes from the sibling flood_it module
from flood_it import Board, Config


# GreedyAgent is a deterministic policy: flood with the color that yields the
# largest connected region on the very next move (ties -> lowest color index)

class GreedyAgent:
    # returns the best color to flood the board with, or None if it is over
    def select_move(self, board: Board):
        # no moves make sense once the game has ended
        if board.is_over():
            return None

        # color currently owned by the flood -> flooding it is a wasted move
        current = board.grid[0][0]

        # track the best color found so far (None -> no candidate yet)
        best_color = None

        # track its resulting coverage (start below any real value)
        best_coverage = -1

        # try every color except the one we already own
        for color in range(board.config.colors):
            # skip the current color so we never waste a turn
            if color == current:
                continue

            # simulate the flood on an independent copy of the board
            sim = board.clone()
            # (flood is guaranteed to succeed: color is valid and differs)
            sim.flood(color)

            # remember this color if it covers more newly-flooded cells
            # ('>' not '>=' keeps the lower index when coverages are tied)
            if sim.coverage > best_coverage:
                best_coverage = sim.coverage
                best_color = color

        # hand back the winning color (None if every move was pointless)
        return best_color


# plays one game of Flood-It to completion using the given agent's policy
def solve(board: Board, agent: GreedyAgent) -> Board:
    # keep making greedy moves until the board is won or moves run out
    while not board.is_over():
        # ask the agent which color to flood with (None = stuck safety net)
        color = agent.select_move(board)
        if color is None:
            break
        # apply the chosen move to the board in place
        board.flood(color)
    # return the finished board so callers can read is_solved/moves_used
    return board


# benchmark harness: run the greedy agent over many random boards
def main():
    # create the command-line argument parser with a short description
    parser = argparse.ArgumentParser(description="Benchmark the greedy solver")
    # register a --size flag, an integer, defaulting to 14
    parser.add_argument("--size", type=int, default=14, help="grid size (default 14)")
    # register a --colors flag, an integer, defaulting to 6
    parser.add_argument("--colors", type=int, default=6,
                        help="number of colors (default 6)")
    # register a --episodes flag, an integer, defaulting to 100
    parser.add_argument("--episodes", type=int, default=100,
                        help="boards to solve (default 100)")
    # register a --seed flag, an integer, defaulting to 0
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    # read the arguments that the user actually passed on the command line
    args = parser.parse_args()

    # reject grid sizes outside the playable range (2..26)
    if not (2 <= args.size <= 26):
        parser.error("size must be between 2 and 26")
    # reject color counts that don't fit in the available palette
    if not (2 <= args.colors <= 8):
        parser.error("colors must be between 2 and 8")
    # reject non-positive episode counts
    if args.episodes < 1:
        parser.error("episodes must be at least 1")

    # build one config; its move_limit derives from size + colors
    config = Config(args.size, args.colors)
    # the agent whose policy we are benchmarking
    agent = GreedyAgent()
    # counters for the aggregate statistics
    solved = 0
    total_moves = 0
    won_moves = 0
    # solve one fresh board per episode
    for i in range(args.episodes):
        # a fresh, independently-seeded RNG makes every episode reproducible
        board = Board(config, rng=Random(args.seed + i))
        # run the greedy policy to the end of the game
        solve(board, agent)
        # accumulate results from the finished game
        total_moves += board.moves_used
        if board.is_solved():
            solved += 1
            won_moves += board.moves_used

    # report the aggregate statistics
    print(f"size={config.size}x{config.size} colors={config.colors} "
          f"move_limit={config.move_limit} episodes={args.episodes}")
    print(f"win rate: {solved}/{args.episodes} "
          f"({100.0 * solved / args.episodes:.1f}%)")
    # average moves across won games only (losses waste the full budget)
    if solved:
        print(f"avg moves (won): {won_moves / solved:.2f}")
    # average moves across every game, wins and losses alike
    print(f"avg moves (all): {total_moves / args.episodes:.2f}")


# only run main() when this file is executed directly, not when imported
if __name__ == "__main__":
    main()