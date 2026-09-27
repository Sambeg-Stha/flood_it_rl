# Import necessary libraries
import numpy as np # For numerical operations (like mean)
import random # For random choices (exploration, tie-breaking)
from collections import defaultdict # Convenient for creating nested dictionaries for memory
from typing import Tuple, Dict, List, DefaultDict, Any # For type hinting
from flood_it import Board, Config
from state_features import feature, bucket, state_key

#file saving
import os
import csv

# Set random seeds for reproducibility
seed: int = 42
random.seed(seed)
np.random.seed(seed)

#e greedy parameter
e_start = 1
e_min = 0.3
e_deacy = 0.999

#iteration parameters
episodes = 500000

#q learning values, 
learn_rate = 0.3
discount_rate = 0.8
LOSS = -1
WIN = 2
WASTE = -2.5

MEMORY = "model_data/q_data_4_8.csv"

#state space as board configuration for 3x3 color 3 board
def state_space(board : Board):
    flood_color = board.grid[0][0]
    flood_cells = board.coverage
    flooded = board.connected()

    outside_counts = {}
    boundary_counts = {}
    for r, row in enumerate(board.grid):
        for c, color in enumerate(row):
            if (r, c) in flooded:
                continue
            outside_counts[color] = outside_counts.get(color, 0) + 1
            
            # cell belongs to the boundary if any neighbor is part of the flood
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if (r + dr, c + dc) in flooded:
                    boundary_counts[color] = boundary_counts.get(color, 0) + 1
                    break  # count each cell once, even if it touches the flood on 2 sides

    best_outside = max(
        range(board.config.colors),
        key=lambda c: (outside_counts.get(c, 0), -c)
    )
    best_boundary = max(
        range(board.config.colors),
        key=lambda c: (boundary_counts.get(c, 0), -c)
    )

    return (flood_color, flood_cells, best_outside, best_boundary, board.moves_left)

#reward enginnering convergence
def reward(board : Board,preconvergence : int, gamma : float) -> float:
    #gain is rewad based on convergance relative to the size of baord
    n = board.config.size ** 2
    if board.is_solved():
        return WIN
    if board.is_over():
        return LOSS
    new = board.coverage / n
    prev = preconvergence / n
    r = -0.05 + gamma * new - prev
    if board.coverage == preconvergence:
        r += WASTE
    return r


#Q-learning
def q_update(memo, state_, action_, reward_, next_state_, action_space, alpha, gamma):
    if state_ not in memo:
        memo[state_] = {}

    max_future = 0.0
    old_q_val = memo[state_].get(action_, 0)
    if next_state_ is not None:
        nxt = memo.get(next_state_, {})
        max_future = max((nxt.get(a2, 0.0) for a2 in range(1, action_space)), default=0.0)
    memo[state_][action_] = old_q_val + alpha * (reward_ + gamma * max_future - old_q_val)
#cvs saves
def save_memory(memory, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "action", "q_value"])
        for state, actions in memory.items():
            for action, q in actions.items():
                writer.writerow([state, action, q])   # state is the packed int key

#load memory from csv
def load_memory(path):
    memo = {}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)                                  # skip header
        for row in reader:
            state = int(row[0])                       # packed int, not a tuple
            action = int(row[1])
            reward = float(row[2])
            if state not in memo:
                memo[state] = {}
            memo[state][action] = reward
    return memo


#choose action:
def choose_action(board, key, memory, epsilon, actions_space):
    remap, inv, boundary_colors, _, _, _ = feature(board)
    valid = [a for a in range(1, actions_space) if a in boundary_colors]
    if not valid:                       
        valid = list(range(1, actions_space))

    if random.random() < epsilon:
        untried = [a for a in valid if a not in memory.get(key, {})]
        picked = random.choice(untried if untried else valid)
        return inv[picked]             

    q = [float("-inf") if a not in valid
        else memory.get(key, {}).get(a, 0.0)
        for a in range(1, actions_space)]
    best = max(q)
    winners = [a for a, v in enumerate(q, start=1) if v == best]
    picked = random.choice(winners)
    return inv[picked]       

def train(config: Config, epi, start, min_e, decay, alpha, gamma, print_every: int = 1000):
    colors = config.colors
    epsilon = start
    train_memory = {}
    for i in range(1, epi + 1):
        board = Board(config)
        while not board.is_over():
            key = state_key(board, board.moves_left)
            action = choose_action(board, key, train_memory, epsilon, colors)
            remap, _, _, _, _, _ = feature(board)
            action_canon = remap[action]                # canonical id stored in table

            prev_cov = board.coverage
            board.flood(action)
            r = reward(board, prev_cov, gamma)

            next_state = None if board.is_over() else state_key(board, board.moves_left)
            q_update(train_memory, key, action_canon, r, next_state, colors, alpha, gamma)

        epsilon = max(min_e, epsilon * decay)
        alpha   = max(0.05, alpha * 0.9999)

        if i % print_every == 0:
            print(f"Ep {i}/{epi} | EPSILON : {epsilon:.3f}")
    return train_memory


#evaluation
def evaluate(memo, config: Config, ep=episodes):
    colors = config.colors
    solved = 0
    total_moves = 0
    won_moves = 0

    for _ in range(ep):
        board = Board(config)
        while not board.is_over():
            state = state_key(board, board.moves_left)   # moves_left read from the board
            action = choose_action(board, state, memo, epsilon=0.0, actions_space=colors)
            board.flood(action)

        total_moves += board.moves_used
        if board.is_solved():
            solved += 1
            won_moves += board.moves_used

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
        state = state_key(board, board.moves_left)
        return choose_action(board, state, self.memo, epsilon=0.0, actions_space=board.config.colors)

def main():
    config : Config = Config(size=4, colors=8)

    trained_memory = train(config, episodes, e_start, e_min, e_deacy,alpha=learn_rate, gamma=discount_rate)
    save_memory(trained_memory, MEMORY)
    evaluate(trained_memory, config, ep= 20000)

if __name__ == "__main__":
    main()