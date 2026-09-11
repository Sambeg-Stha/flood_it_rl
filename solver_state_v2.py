# Import necessary libraries
import numpy as np # For numerical operations (like mean)
import random # For random choices (exploration, tie-breaking)
from collections import defaultdict # Convenient for creating nested dictionaries for memory
from typing import Tuple, Dict, List, DefaultDict, Any # For type hinting
from flood_it import Board, Config

#file saving
import os
import csv

# Set random seeds for reproducibility
seed: int = 42
random.seed(seed)
np.random.seed(seed)

#e greedy parameter
e_start = 1
e_min = 0.01
e_deacy = 0.995

#iteration parameters
episodes = 500000

MEMORY = "model_data/simple_rl_agent_state_v2_3x3_8.csv"

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

    return (flood_color, flood_cells, best_outside, best_boundary)

#reward enginnering convergence
def reward(board : Board,preconvergence : int) -> float:
    gain : int = board.coverage - preconvergence
    #reward is based on how much board is flooded compared to the last move
    r : float = float(gain)
    if board.is_solved():
        r += 10.0
    return r


#memory of agent
agent_memory = {}
def update_memory(memory, state, action, reward):
    if state not in memory:
        memory[state] = {}
    if action not in memory[state]:
        memory[state][action] = []
    memory[state][action].append(reward)

#cvs saves
def save_memory(memory, path):
    with open(path, "w", newline= "") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "action", "reward(average)"])
        for state, action in memory.items():
            for action, reward_list in action.items():
                avg = sum(reward_list)/len(reward_list)
                state_str = "|".join(str(x) for x in state)
                writer.writerow([state_str, action, avg])

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

            memo[state_key][action] = [reward]
    return memo


#choose action:
def choose_action(board:Board, state, memory, epsilon, actions_space):
    current_color : int = board.grid[0][0]

    if random.random() < epsilon:
        valid = []
        for c in range(actions_space):
            if c != current_color:
                valid.append(c)
        return random.choice(valid)
    else:
        avg_reward = []
        for a in range(actions_space):
            if a == current_color:
                avg_reward.append(float("-inf"))
                continue
            reward_list = memory.get(state, {}).get(a,[])
            if reward_list:
                avg_reward.append(sum(reward_list)/len(reward_list))
            else:
                avg_reward.append(0.0)
        best = max(avg_reward)
        best_action = [a for a,v in enumerate(avg_reward) if v == best]
        return random.choice(best_action)

def train(config: Config, epi, start, min_e, decay, print_every : int = 1000):
    action_space = config.colors
    epsilon = start
    #fresh memory
    train_memory = {}
    for i in range(1, epi + 1):
        board = Board(config)
        state = state_space(board)

        while not board.is_over():
            action = choose_action(board, state, train_memory, epsilon, action_space)
            prev_cov = board.coverage
            board.flood(action)
            r = reward(board, prev_cov)

            #updating the agent memeory
            update_memory(train_memory, state, action, r)
            state = state_space(board)
        
        epsilon = max(min_e, epsilon * decay)

        if i % print_every == 0 :
            print(f"Ep {i}/{epi} | EPSILION : {epsilon:.3f}")
    return train_memory


#evaluation
def evaluate(memo, config : Config, ep = episodes):
    action_space = config.colors
    solved = 0
    total_moves = 0
    won_moves = 0

    for _ in range(ep):
        board = Board(config)
        while not board.is_over():
            state = state_space(board)
            action = choose_action(board, state, memo, epsilon= 0.0, actions_space=action_space)
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
class RLagent:
    def __init__(self):
        if not os.path.exists(MEMORY):
            raise FileNotFoundError(f"{MEMORY} not found")
        self.memo = load_memory(MEMORY)

    def select_move(self, board : Board):
        if board.is_over():
            return None
        state_key = state_space(board)
        return choose_action(board, state_key, self.memo, epsilon= 0.0, actions_space=board.config.colors)

def main():
    config : Config = Config(size=3, colors=8)

    trained_memory = train(config, episodes, e_start, e_min, e_deacy)
    save_memory(trained_memory, MEMORY)
    evaluate(trained_memory, config, ep= 10000)

if __name__ == "__main__":
    main()