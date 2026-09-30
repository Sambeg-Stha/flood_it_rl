# Flood-It

A playable Flood-It game with a tkinter interface, plus two kinds of computer
player: a simple greedy heuristic, and a Q-learning agent trained on board sizes
2x2 through 5x5.

## Quick start

Play the game:

```bash
python main.py                          # 14x14 board, 6 colors
python main.py --size 4 --colors 8      # 4x4 board, 8 colors (uses the trained agent)
```

Train an agent (each file trains one board size):

```bash
python q_4_8.py     # trains the 4x4 / 8-color agent
python q_5_8.py     # trains the 5x5 / 8-color agent
```

Check how good a trained agent is:

```bash
python q_4_8.py       # training runs first, then reports the final score
```

Training writes its table to `model_data/`. The GUI reads that same file, so
train before playing if you want the agent to drive the Solve button.

---

## Files

| File | What it does |
|------|--------------|
| `flood_it.py` | The game itself: `Config` (board size, colors, move limit) and `Board` (the grid, flooding, win check). Everything else builds on this. |
| `main.py` | The tkinter game window. Wires the board to the screen, the mouse, and the Solve button. |
| `solver_method.py` | `GreedyAgent` — the heuristic player. Also has a standalone benchmark you can run. |
| `q_2_8.py` `q_3_8.py` `q_4_8.py` `q_5_8.py` | The trained Q-learning agents, one per board size. |
| `q_learning.py` | A small measurement script (not an agent). See [Checking the state space](#checking-the-state-space). |
| `q_solver.py` | The original prototype agent. Kept for reference only — see [Known issues](#known-issues). |

The four `q_*_8.py` files are **identical copies**. They differ only in three
lines: the table filename, the class name, and the board size. To add a new size,
copy `q_4_8.py`, change those three lines, and run it.

---

## How the move limit is calculated

You do not set the move limit yourself. `Config` derives it from the board size
and color count, so every difficulty is roughly fair.

The formula comes from the paper *The Complexity of Flood Filling Games*. It
builds a recommended interval:

- lower bound: `sqrt(colors - 1) * size / 2 - colors / 2`
- upper bound: `2 * size + sqrt(2 * colors) * size + colors`

Then it takes the midpoint and cuts it by `decrement_percentage` (37%, set at the
top of `flood_it.py`). Raising or lowering that one number makes the whole game
easier or harder everywhere.

---

## The greedy player

`GreedyAgent.select_move(board)` does the simplest sensible thing: copy the board,
try flooding with every color except the one you already own, and pick whichever
one covers the most new cells. One move costs a handful of cheap simulations, and
it never makes a bad move in the sense of wasting the turn.

It is a surprisingly strong baseline, which matters more than it sounds — see
[Results](#results).

You can benchmark it on its own:

```bash
python solver_method.py --size 5 --colors 8 --episodes 100
```

---

## How the Q-learning agent works

### The reward

Each move is scored after the fact:

| What happened | Reward |
|---|---|
| Board fully flooded | `+10` |
| Out of moves, not solved | `-15` |
| Made progress | `-0.05 + 10 × (cells gained / total cells)` |
| Gained nothing (wasted the turn) | the above, plus `-10` |

The progress reward is scaled by `SHAPE` (currently `10.0`). That number matters
more than it looks: if it is too low, the agent only learns "don't waste a turn"
and cannot tell a good move from a merely acceptable one. If it is too high, the
agent chases big immediate captures and ignores whether it will actually finish.

`WASTE = -10` is deliberately harsh. On a 10-move board one wasted turn costs
half the total progress reward available in a whole game.

### The state

This is the part that took the most work, and the part most likely to break
silently.

A Q-table is just a number stored per (state, action) pair. Those numbers are only
trustworthy if each pair gets visited a reasonable number of times. So the state
needs to describe the board well enough to tell positions apart, but not so
finely that every board gets its own private entry.

The current state has four numbers:

```
(cells flooded, best possible gain, second best gain, moves remaining)
```

**"Best possible gain"** is the real prize: the agent copies the board, floods with
each color, and counts how many cells it would win. That number carries more
information than anything cheaper — it knows the difference between one color
offering a six-cell blob and the same color appearing as five disconnected
single cells, which look identical to any count-based shortcut.

### Colour canonicalization — the fix that mattered

This is worth understanding, because the first version of the new state failed
badly (0.5% win rate) and the reason is instructive.

The naive version of the state included *how many* cells each color offered but
not *which* color. That looks fine, and it isn't. Consider two boards that both
have one color offering a 4-cell gain and all others offering nothing — one where
the good color is red, one where it's blue. Under that state, both boards collapse
into the **same key**. So `Q[state][red]` gets trained partly from boards where red
is the great move and partly from boards where red is useless. Average them and you
get mush.

That is exactly what happened. The trained agent picked the genuinely best move
only **16.8%** of the time — barely better than random — and wasted **59.5%** of
its turns. It had learned "every color is mediocre," which is true on average and
useless in practice.

The fix is to stop treating colors as fixed labels. Before building the state, the
board is relabelled into a canonical order:

| Canonical label | Meaning |
|---|---|
| `0` | the color you already own (not a legal move) |
| `1` | the biggest gain |
| `2` | second biggest gain |
| `3`–`7` | the rest |

Now canonical action `1` **always** means "the best move," in every board, forever.
`Q[state][1]` is unambiguous. And because the labels no longer carry real color
information, the state doesn't need a color field at all.

The table gets dramatically smaller and denser, because every state now pools what
it learned across all color arrangements:

| Board | States before | States after | Actions learned per state before → after |
|-------|---------------|--------------|--------------------------------------------|
| 4x4 | 42,537 | **1,547** | 2.95 → **3.99** |
| 5x5 | — | **4,145** | — → **3.86** |

Agents are trained and saved per size, since the state includes absolute cell
counts and the move limit differs by size.

### Training setup

| Setting | 2x2 / 3x3 / 4x4 | 5x5 | Notes |
|---|---|---|---|
| `discount_rate` | 0.9 | 0.95 | 5x5 gets 12 moves vs 10, so it discounts less |
| `learn_rate` | 0.3 | 0.3 | Constant — deliberately not decayed |
| `e_min` | 0.05 | 0.05 | Floor of the exploration rate |
| `episodes` | 500,000 | 500,000 | |

Three training details that are easy to get wrong:

**Exploration decays fast and then stops.** Epsilon falls as `0.999^n` and bottoms
out at `e_min` after roughly 800–3,000 episodes. It is *not* a gradual fade across
the whole run. An earlier version sat at `e_min = 0.45` for 99.8% of training,
meaning nearly half of everything the table ever learned came from random moves.

**The learning rate is not decayed.** A decaying alpha is meant for states you
revisit millions of times. These tables see each entry a handful of times, so
alpha stays high.

**Discounts are checked against the move limit, not chosen by feel.** A win is
worth `+10` but arrives at the very end. With gamma `0.8` on a 10-move board, the
value of a future win at move 1 is `0.8¹¹ ≈ 0.09` — the agent literally cannot
tell a winnable position from a lost one. At `0.95` that rises to `0.57`.

### How a move is chosen

1. If we are still exploring, prefer a move this state has never tried, else pick
   at random.
2. Otherwise take the move with the highest learned value.
3. If the table knows nothing about this state, or two moves tie, fall back to the
   best immediate gain.

Step 3 matters more than it should. Because the table is sparse, plenty of states
have never been seen, and an unseen state used to mean "play randomly" — which
showed up directly as lost games.

---

## Checking the state space

`q_learning.py` answers the question *"is this table getting enough visits per
state to learn anything?"* without training anything. It plays 2,000 boards using
the greedy policy and counts how many distinct states appear and how often each is
revisited.

```bash
python q_learning.py
```

Two numbers come out. `distinct states` should be well under the theoretical
maximum — if the agent only ever reaches a small slice, a wide state space costs
you nothing. `avg visits/state` should be comfortably above about 20; below 8, the
state has more features than the training data can support, and the fix is to
round the gain numbers into buckets like `0 / 1 / 2-3 / 4+` instead of using exact
counts.

---

## Results

Measured on 2,000 held-out boards that were never used for training or for picking
a checkpoint (seed 999):

| Board | Move limit | Greedy | Q-table | States | Actions/state |
|-------|-----------|--------|---------|--------|----------------|
| 2x2 | 5 | 100.0% | 100.0% | 35 | 6.97 |
| 3x3 | 7 | 96.7% | 96.7% | 400 | 5.05 |
| 4x4 | 10 | 88.9% | 86.8% | 1,547 | 3.99 |
| 5x5 | 12 | 71.7% | 64.8% | 4,145 | 3.86 |

**Read this honestly: the Q-table is currently matching the greedy heuristic on the
small boards and losing to it on the larger ones.** It is not yet outperforming
the thing it was built on top of.

That is the open problem, not a finished result. Two caveats on the numbers:

- These tables came from partial training runs, not the full 500,000 episodes. The
  4x4 table in particular is small enough to suggest an early checkpoint was the
  best one found.
- The gap is small enough to partly be sampling noise, but the direction is
  consistent across both large sizes, so it is probably real.

The next things worth trying, roughly in order:

1. **Finish the training runs.** Both large tables are from incomplete runs.
2. **Sweep `SHAPE`.** The progress reward weight is the least-tuned number in the
   reward and the most likely to be mistuned.
3. **Add the missing diagnostic.** Measure how often the agent picks the genuinely
   best move and how often it wastes a turn. This is what caught the
   canonicalization bug — it reported 16.8% and 59.5% when the agent was
   effectively random. Those two numbers are a better health check than win rate
   alone.
4. **Resolve the tie-breaking shortcut.** Ties in gain are currently broken by real
   color index, which is not fully invariant under relabeling, leaving a little
   residual mixing. Breaking ties on a board property instead (such as where on the
   board the first captured cell sits) would close it.

---

## Known issues

**`save_memory` does not create its directory.** `load_memory` does
(`q_*_8.py`, in `load_memory`). If `model_data/` is ever cleaned out, saving fails
at the first checkpoint.

---

## Model data

`model_data/` holds the learned tables as CSV, one file per board size, named
`q_data_<size>_<colors>_v3.csv`. Each row is `state|action|q_value`.

The `_v3` suffix marks tables built with the canonicalized four-number state.
Earlier tables (`q_data_4_8.csv`, `q_data_5_8.csv`) use the older five-number state
and **cannot be loaded by the current agents** — the keys mean different things.
They are kept only for comparison.

Every agent class raises `FileNotFoundError` at startup if its table is missing,
so train before opening the GUI at a size you have not trained yet.
