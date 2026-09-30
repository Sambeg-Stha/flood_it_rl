# Import necessary libraries
import numpy as np # For numerical operations (like mean)
import random # For random choices (exploration, tie-breaking)
from collections import defaultdict # Convenient for creating nested dictionaries for memory
from typing import Tuple, Dict, List, DefaultDict, Any # For type hinting
from flood_it import Board, Config
from random import Random

#file saving
import os
import csv

# Set random seeds for reproducibility
seed: int = 42
EVAL_SEED = 1234
FINAL_SEED = 999
random.seed(seed)
np.random.seed(seed)

#e greedy parameter
e_start = 1
e_min = 0.05
e_deacy = 0.999

#iteration parameters
episodes = 500000

#q learning values, 
learn_rate = 0.3
discount_rate = 0.9
LOSS = -15
WIN = 10
WASTE = -10
SHAPE = 10.0     # scales the progress reward
STEP = -0.05     # small cost per move, discourages long routes

MEMORY = "model_data/q_data_4_8_v3.csv"

# canonical colours: 0 = the one we own, 1 = biggest gain, 2 = next, then the rest.
# returns (real colour for each canonical label, gains sorted best first)
def canonical_view(board):
    current = board.grid[0][0]
    now = board.coverage
    valid = [c for c in range(board.config.colors) if c != current]

    gains = []
    for color in valid:
        sim = board.clone()
        sim.flood(color)
        gains.append(sim.coverage - now)

    # biggest gain first; ties keep the lower colour index ahead
    order = sorted(range(len(valid)), key=lambda i: (-gains[i], valid[i]))

    to_real = [current]
    for i in order:
        to_real.append(valid[i])

    return to_real, [gains[i] for i in order]

#state space as board configuration for 3x3 color 3 board
def state_space(board : Board):
    to_real, ordered = canonical_view(board)
    return (board.coverage, ordered[0], ordered[1], board.moves_left), to_real

#reward enginnering convergence
def reward(board : Board, preconvergence : int) -> float:
    n = board.config.size ** 2
    if board.is_solved():
        return WIN
    if board.is_over():
        return LOSS
    gain = (board.coverage - preconvergence) / n
    r = STEP + SHAPE * gain
    if board.coverage == preconvergence:
        r += WASTE
    return r


#Q-learning
def q_update(memo, state_, action_, reward_, next_state_, action_space, alpha ,gamma):
    if state_ not in memo:
        memo[state_] = {}
    
    #for next state terminal
    max_future = 0.0
    old_q_val = memo[state_].get(action_, 0)
    if next_state_ is not None:
        row = memo.get(next_state_, {})
        # only consider actions that have actually been visited
        known = [row[a] for a in range(1, action_space) if a in row]
        max_future = max(known) if known else 0.0
    memo[state_][action_] = old_q_val + alpha * (reward_ + gamma * max_future - old_q_val)
#cvs saves
def save_memory(memory, path):
    with open(path, "w", newline= "") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "action", "q_value"])
        for state, actions in memory.items():
            state_str = "|".join(str(x) for x in state)
            for action, q in actions.items():
                writer.writerow([state_str, action, q])

#load memory from csv
def load_memory(path):
    memo = {}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "r") as f:
        reader = csv.reader(f)
        #skiping header like state|action|reward
        next(reader)
        for row in reader:
            state_key = tuple(int(x) for x in row[0].split("|"))
            action = int(row[1])
            reward = float(row[2])
            if state_key not in memo:
                memo[state_key] = {}

            memo[state_key][action] = reward
    return memo


# picks the color with the highest gain; valid colors in color order
def best_gain_color(board, valid, gains):
    best = valid[0]
    best_gain = -1
    for i, color in enumerate(valid):
        if gains[i] > best_gain:
            best_gain = gains[i]
            best = color
    return best

#choose action:
def choose_action(state, memory, epsilon, actions_space):
    row = memory.get(state, {})

    if random.random() < epsilon:
        untried = [a for a in range(1, actions_space) if a not in row]
        if untried:
            return random.choice(untried)   # try every action per state, once
        return random.choice(range(1, actions_space))

    known = [a for a in range(1, actions_space) if a in row]
    if not known:
        return 1                   # canonical 1 is always the biggest gain

    best = max(row[a] for a in known)
    tied = [a for a in known if row[a] == best]
    return min(tied)

def train(config: Config, epi, start, min_e, decay, alpha, gamma, eval_grids=None, print_every: int = 1000, eval_every: int = 25000):
    action_space = config.colors
    epsilon = start
    #fresh memory
    train_memory = {}
    best_score = -1.0
    for i in range(1, epi + 1):
        board = Board(config, rng=Random(seed * 1000003 + i))
        state, to_real = state_space(board)

        while not board.is_over():
            action = choose_action(state, train_memory, epsilon, action_space)
            prev_cov = board.coverage
            board.flood(to_real[action])
            r = reward(board, prev_cov)

            if board.is_over():
                q_update(train_memory, state, action, r, None, action_space, alpha, gamma)
            else:
                next_state, to_real = state_space(board)
                q_update(train_memory, state, action, r, next_state, action_space, alpha, gamma)
                state = next_state

        epsilon = max(min_e, epsilon * decay)


        if i % print_every == 0 :
            print(f"Ep {i}/{epi} | EPSILION : {epsilon:.3f}")

        if eval_grids and i % eval_every == 0:
            score = win_rate(train_memory, config, eval_grids)
            mark = "  <- best" if score > best_score else ""
            print(f"   eval @ {i}: {100 * score:.1f}%{mark}")
            if score > best_score:
                best_score = score
                save_memory(train_memory, MEMORY)
    return best_score

# fixed eval puzzles, so every checkpoint is scored on identical boards
def make_eval_grids(config, count, base):
    return [Board(config, rng=Random(base + i)).grid for i in range(count)]

# plays one grid to the end with the greedy policy
def play(memo, config, grid):
    board = Board(config, grid=[row[:] for row in grid])
    while not board.is_over():
        state, to_real = state_space(board)
        action = choose_action(state, memo, epsilon=0.0, actions_space=config.colors)
        board.flood(to_real[action])
    return board

# just the win rate, used as the checkpoint score
def win_rate(memo, config, grids):
    solved = 0
    for grid in grids:
        if play(memo, config, grid).is_solved():
            solved += 1
    return solved / len(grids)

#evaluation
def evaluate(memo, config, grids):
    solved = 0
    total_moves = 0
    won_moves = 0

    for grid in grids:
        board = play(memo, config, grid)
        total_moves += board.moves_used
        if board.is_solved():
            solved += 1
            won_moves += board.moves_used

    ep = len(grids)
    print(f"size={config.size}x{config.size} colors={config.colors} "
          f"move_limit={config.move_limit} eval_episodes={ep}")
    print(f"win rate: {solved}/{ep} ({100.0 * solved / ep:.1f}%)")
    if solved:
        print(f"avg moves (won): {won_moves / solved:.2f}")
    print(f"avg moves (all): {total_moves / ep:.2f}")

#RL AGENT  FOR MAIN GAME
class Q_4_8:
    def __init__(self):
        if not os.path.exists(MEMORY):
            raise FileNotFoundError(f"{MEMORY} not found")
        self.memo = load_memory(MEMORY)

    def select_move(self, board : Board):
        if board.is_over():
            return None
        state, to_real = state_space(board)
        action = choose_action(state, self.memo, epsilon= 0.0, actions_space=board.config.colors)
        return to_real[action]

def main():
    config : Config = Config(size=4, colors=8)

    # checkpoints are scored on these fixed boards
    eval_grids = make_eval_grids(config, 2000, EVAL_SEED)

    best_score = train(config, episodes, e_start, e_min, e_deacy,
                       alpha=learn_rate, gamma=discount_rate,
                       eval_grids=eval_grids)
    print(f"best checkpoint win rate: {100 * best_score:.1f}%")

    # final report uses a DIFFERENT seed: picking the best of 20 checkpoints on
    # eval_grids means eval_grids is now a selection set, not a clean test set
    evaluate(load_memory(MEMORY), config, make_eval_grids(config, 20000, FINAL_SEED))

if __name__ == "__main__":
    main()