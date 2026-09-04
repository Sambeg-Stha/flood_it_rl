# Import necessary libraries
import numpy as np # For numerical operations (like mean)
import random # For random choices (exploration, tie-breaking)
from collections import defaultdict # Convenient for creating nested dictionaries for memory
from typing import Tuple, Dict, List, DefaultDict, Any # For type hinting
from flood_it import Board, Config

# Set random seeds for reproducibility
seed: int = 42
random.seed(seed)
np.random.seed(seed)

#e greedy parameter
e_start = 1
e_min = 0.01
e_deacy = 0.995

#iteration parameters
episodes = 20000

#state space as board configuration for 3x3 color 3 board
def state_space(board : Board):
    flat = []
    for row in board.grid:
        for cell in row:
            flat.append(cell)
    
    flat.append(board.moves_used)
    return tuple(flat)

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

#choose action:
def choose_action(board:Board, state, memory, epsilion, actions_space):
    current_color : int = board.grid[0][0]

    if random.random() < epsilion:
        valid = []
        for c in range(actions_space):
            if c != current_color:
                valid.append(c)
        return random.choice(valid)
    else:
        avg_reward = []
        for a in random(actions_space):
            if a == current_color:
                avg_reward.append(float("-inf"))
                continue
            reward = memory[state_space][a]
            if reward:
                avg_reward.append(sum(reward)/len(reward))
            else:
                avg_reward.append(0.0)
        best = max(avg_reward)
        best_action = [a for a,v in enumerate(avg_reward) if v == best]
        return random.choice(best_action)

def train(config: Config, epi, start, min_e, decay, print_every : int = 1000):
    action_space = config.colors
    epsilion = start
    #fresh memory
    train_memory = {}
    for i in range(1, epi + 1):
        board = Board(config)
        state = state_space(board)

        while not board.is_over():
            action = choose_action(board, state, train_memory, epsilion, action_space)
            prev_cov = board.coverage
            r = reward(board, prev_cov)

            #updating the agent memeory
            update_memory(train_memory, state, action, r)
            state = state_space(board)
        
        epsilion = max(min_e, epsilion * decay)

        if i % print_every == 0 :
            print(f"Ep {i}/{epi} | EPSILION : {epsilion:.3f}")
    return train_memory


#evaluation
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
            action = choose_action(board, state, memo, epsilion= 0.0, actions_space=action_space)
            board.flood(action)

        total_moves += board.moves_used
        if board.is_solved():
            solved += 1
            won_moves += board.moves_used
    print(f"size={config.size}x{config.size} colors={config.colors} "
          f"move_limit={config.move_limit} eval_episodes={episodes}")
    print(f"win rate: {solved}/{episodes} ({100.0 * solved / episodes:.1f}%)")
    if solved:
        print(f"avg moves (won): {won_moves / solved:.2f}")
    print(f"avg moves (all): {total_moves / episodes:.2f}")

def main():
    config : Config = Config(size=3, colors=3)

    trained_memory = train(config, episodes, e_start, e_min, e_deacy)
    evaluate(trained_memory, config, ep= 100)

if __name__ == "__main__":
    main()