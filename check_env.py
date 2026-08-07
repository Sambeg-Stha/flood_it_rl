# check_env.py - a tiny harness to prove the FloodItEnv is wired up correctly.
#
# It runs the environment start-to-finish with a random agent and prints the
# key values at every step, so you can see exactly what the RL interface hands
# an agent (observation, reward, done flag, and the extra info dict).
#
# The seed controls the randomness:
#   - Same seed  -> same board, same moves, same reward (reproducible).
#   - No seed    -> a fresh random board every run.
#
# Usage (from the project folder):
#     .\venv\Scripts\python.exe .\check_env.py              # random seed
#     .\venv\Scripts\python.exe .\check_env.py 7            # fixed seed 7
#     .\venv\Scripts\python.exe .\check_env.py 7 3          # 3 episodes, seed 7

# 'argv' lets us read the command-line arguments (seed, episode count)
from sys import argv

# import the environment we built; it imports the game logic from flood_it.py
from env import FloodItEnv


def play_one_episode(env, seed):
    # start a brand-new episode with the given (or random) seed
    obs, info = env.reset(seed=seed)
    print("=== Initial state (seed=" + str(seed) + ") ===")
    print("observation:", obs)
    print("info:", info)
    print()

    # play the whole game with a random agent and report each step
    print("=== Rollout (random agent) ===")
    print(f"{'step':>4} | {'action':>6} | {'reward':>6} | {'moves_left':>10} "
          f"| {'flooded':>7} | done")
    total_reward = 0.0
    step = 0
    while not env.done:
        # pick a random legal color from the current board's rng
        action = env.action_space.sample(env.rng)
        # send the action and collect the results
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        print(f"{step:>4} | {action:>6} | {reward:>6.1f} | {info['moves_left']:>10} "
              f"| {info['flooded_cells']:>7} | {done}")

    # summarize the episode
    print()
    print("=== End of episode ===")
    print("steps taken:", step)
    print("total reward:", round(total_reward, 1))
    # 'done' alone doesn't say if we won - check it directly from the board
    print("solved (won):", env.board.is_solved())
    print("final board:")
    env.render()
    print()


def main():
    # a small board so printed grids stay readable (6x6, 3 colors, 25 moves)
    env = FloodItEnv(size=14, colors=8, move_limit=25)

    # parse optional command-line arguments
    base_seed = int(argv[1]) if len(argv) > 1 else None  # seed, default random
    episodes = int(argv[2]) if len(argv) > 2 else 1      # how many to run

    # run the requested number of episodes, each with a fresh seed
    for i in range(episodes):
        # if a base seed was given, bump it per episode so each one differs
        seed = (base_seed + i) if base_seed is not None else None
        play_one_episode(env, seed)
        # mark the episode done so the next reset() must be issued by the loop
        env.done = True


# run main() only when the file is executed directly (not when imported)
if __name__ == "__main__":
    main()