from q_4_8 import *
import time
from collections import Counter

config = Config(size=4, colors=8)
c = Counter()
t0 = time.time()

for i in range(2000):
    board = Board(config, rng=Random(i))
    state, to_real = state_space(board)
    while not board.is_over():
        c[state] += 1
        board.flood(to_real[1])
        if not board.is_over():
            state, to_real = state_space(board)

print(f"{time.time() - t0:.1f}s for 2000 episodes")
print(f"distinct states: {len(c)}")
print(f"avg visits/state: {sum(c.values()) / len(c):.1f}")