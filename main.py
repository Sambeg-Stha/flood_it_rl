# Flood-It game: tkinter user interface

# argparse handles the command-line flags like --size and --colors
import argparse
# tkinter is Python's built-in GUI toolkit, imported under the short name 'tk'
import tkinter as tk

# import the game logic classes from the sibling flood_it module
from flood_it import Board, Config
# import the greedy policy used by the Solve button and the best-move hint
from solver_method import GreedyAgent
from q_solver import RLagent

# list of the actual hex color codes that back the palette (index = color number)
PALETTE = [
    "#e74c3c",  # color 0: red
    "#2ecc71",  # color 1: green
    "#3498db",  # color 2: blue
    "#f1c40f",  # color 3: yellow
    "#9b59b6",  # color 4: purple
    "#e67e22",  # color 5: orange
    "#1abc9c",  # color 6: teal
    "#ecf0f1",  # color 7: off-white
    "#e84393",  # color 8: pink
    "#00cec9",  # color 9: cyan
    "#fab1a0",  # color 10: salmon
    "#636e72",  # color 11: grey
]

# dark gray used for the window background and the gaps between board cells
GRID_BG = "#2d3436"

# human-readable names for the palette, matching PALETTE index-by-index
COLOR_NAMES = [
    "Red", "Green", "Blue", "Yellow", "Purple", "Orange",
    "Teal", "Off-White", "Pink", "Cyan", "Salmon", "Grey",
]

# text color used on top of the best-move hint swatch
HINT_FG = "#1a1a1a"


# Game wires the Board logic up to the tkinter widgets
class Game:
    # constructor: takes the tkinter root window and the game configuration
    def __init__(self, root: tk.Tk, config: Config):
        # remember the root window so other methods can use it
        self.root = root
        # remember the configuration (size, colors, move limit)
        self.config = config
        # set the text shown in the window's title bar
        root.title("Flood It")
        # stop the user from resizing the window and distorting the board
        root.resizable(False, False)

        # create a top frame that holds the status text and buttons
        self.top = tk.Frame(root, bg=GRID_BG)
        # pack it along the top edge, stretched full width, with padding
        self.top.pack(fill="x", padx=8, pady=6)

        # label that displays the move count / win-lose message
        self.moves_label = tk.Label(
            self.top, text="", font=("Segoe UI", 14, "bold"),
            fg="white", bg=GRID_BG,
        )
        # place the label on the left side of the top frame
        self.moves_label.pack(side="left")

        # label that shows the greedy policy's recommended next move
        self.hint_label = tk.Label(
            self.top, text="Best next move: -", font=("Segoe UI", 11, "bold"),
            fg=HINT_FG, bg=GRID_BG, padx=6,
        )
        # place the hint to the right of the move counter, separated by gaps
        self.hint_label.pack(side="left", padx=(24, 0))

        # button that starts a fresh game when clicked
        self.new_button = tk.Button(
            self.top, text="New Game", font=("Segoe UI", 12),
            command=self.new_game,
        )
        # place the button on the right side of the top frame
        self.new_button.pack(side="right")

        # button that opens the settings dialog to adjust the board
        self.settings_button = tk.Button(
            self.top, text="Settings", font=("Segoe UI", 12),
            command=self.open_settings,
        )
        # place the settings button to the left of the New Game button
        self.settings_button.pack(side="right")

        # button that auto-solves the current board with the greedy agent
        self.solve_button = tk.Button(
            self.top, text="Solve", font=("Segoe UI", 12),
            command=self.solve,
        )
        # place the solve button to the left of the settings button
        self.solve_button.pack(side="right", padx=(0, 8))

        # canvas widget where the board squares get drawn
        self.canvas = tk.Canvas(
            root, bg=GRID_BG,
            highlightthickness=0,
        )
        # place the canvas below the top frame with padding
        self.canvas.pack(padx=8, pady=(0, 8))
        # make a left mouse click on the canvas trigger the on_click handler
        self.canvas.bind("<Button-1>", self.on_click)

        # frame that holds the row of clickable color swatches
        self.palette = tk.Frame(root, bg=GRID_BG)
        # place the palette below the board
        self.palette.pack(pady=(0, 8))

        # list where we store each swatch button (kept in case we need them)
        self.swatches = []
        # draw the color swatches for this game's palette
        self._build_palette()

        # size the canvas and set per-cell pixels for the current config
        self._apply_board_size()

        # placeholder for the active board (real one created below)
        self.board = None
        # map from (row, col) to the canvas rectangle id of each drawn cell
        self.rects = {}
        # the greedy policy behind the Solve button and the best-move hint
        if self.config.size == 3:
            print("USING RL AGENT IN THE MAIN GAME")
            self.agent = RLagent()
        else:
            self.agent = GreedyAgent()
        # id of the pending after() step that drives the solve animation
        self.solve_job = None
        # milliseconds to wait between animated solver moves
        self.solve_delay = 200
        # start the first game immediately
        self.new_game()

    # (re)create one swatch button for every color in the current config
    def _build_palette(self):
        # remove any swatches left over from a previous configuration
        for widget in self.palette.winfo_children():
            widget.destroy()
        # reset the stored swatch list before filling it again
        self.swatches = []
        # create one swatch button for every color in this game's palette
        for color in range(self.config.colors):
            # build a small colored square button for this color
            btn = tk.Button(
                self.palette, bg=PALETTE[color],
                width=3, height=1, relief="flat",
                # clicking it floods the board with that color (c default-captured)
                command=lambda c=color: self.pick(c),
            )
            # lay the swatches out in a single row, one column apart
            btn.grid(row=0, column=color, padx=3)
            # remember the button in our list
            self.swatches.append(btn)

    # recompute the per-cell pixel size and resize the canvas to fit the board
    def _apply_board_size(self):
        # pixel size of each board cell; shrinks for larger boards
        self.cell_size = 560 // max(10, self.config.size)
        # size the canvas to exactly fit size*size cells
        self.canvas.config(width=self.cell_size * self.config.size,
                           height=self.cell_size * self.config.size)

    # pop up a modal dialog to change board size, colors, and move limit
    def open_settings(self):
        # temporary window layered on top of the main window
        dialog = tk.Toplevel(self.root)
        dialog.title("Board Settings")
        # keep the dialog attached to the main window and make it modal
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.configure(bg=GRID_BG)
        # prevent clicks from reaching the main window while the dialog is open
        dialog.grab_set()

        # variables bound to the input widgets, pre-filled from current config
        size_var = tk.IntVar(value=self.config.size)
        colors_var = tk.IntVar(value=self.config.colors)

        # label that reports validation errors (hidden until something is wrong)
        error = tk.Label(dialog, text="", fg="#e74c3c", bg=GRID_BG,
                         font=("Segoe UI", 10))
        error.grid(row=0, column=0, columnspan=2, pady=(10, 0))

        # helper that adds one labeled spinbox row to the dialog
        def add_field(row, label, variable, low, high):
            tk.Label(dialog, text=label, fg="white", bg=GRID_BG,
                     font=("Segoe UI", 12)).grid(row=row, column=0,
                                                 sticky="e", padx=(12, 8), pady=6)
            tk.Spinbox(dialog, from_=low, to=high, textvariable=variable,
                       width=6, font=("Segoe UI", 12)).grid(row=row, column=1,
                                                            sticky="w", padx=(0, 12), pady=6)

        # fields: only size and colors; the move limit is auto-calculated
        add_field(1, "Size:", size_var, 2, 26)
        add_field(2, "Colors:", colors_var, 2, 8)

        # applies the chosen values and rebuilds the game, or reports an error
        def apply_settings():
            # spinboxes allow typing, so clamp entries back into valid ranges
            try:
                size = max(2, min(26, size_var.get()))
                colors = max(2, min(8, colors_var.get()))
            except tk.TclError:
                # non-numeric text was typed into a field
                error.config(text="Please enter valid numbers.")
                return
            # commit the new configuration; build a fresh Config so move_limit
            # is re-derived from the new size/colors (since it only computes
            # at construction time)
            self.config = Config(size=size, colors=colors)
            if self.config.size == 3:
                print("USING RL AGENT IN THE MAIN GAME")
                self.agent = RLagent()
            else:
                self.agent = GreedyAgent()
            # resize the board, rebuild the color swatches, and start fresh
            self._apply_board_size()
            self._build_palette()
            self.new_game()
            # close the settings dialog
            dialog.destroy()

        # button frame holding OK and Cancel
        buttons = tk.Frame(dialog, bg=GRID_BG)
        buttons.grid(row=3, column=0, columnspan=2, pady=(6, 12))
        tk.Button(buttons, text="OK", font=("Segoe UI", 11), width=8,
                  command=apply_settings).pack(side="left", padx=6)
        tk.Button(buttons, text="Cancel", font=("Segoe UI", 11), width=8,
                  command=dialog.destroy).pack(side="left", padx=6)

        # wait for the dialog to be drawn, then center it over the main window
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 3
        dialog.geometry(f"+{x}+{y}")

# starts a fresh game: new board, redrawn, status reset
    def new_game(self):
        # cancel any in-progress solve animation before anything else
        if self.solve_job is not None:
            self.root.after_cancel(self.solve_job)
            self.solve_job = None
            # re-enable the buttons that the solver had disabled
            self._solve_done()
        # generate a brand new random Board using the stored configuration
        self.board = Board(self.config)
        # redraw the canvas so it shows the new board
        self.draw()
        # refresh the move counter / status label
        self.update_status()
        # refresh the greedy best-move hint (a solve may have ended it)
        self.update_hint()

    # draws the current board state onto the canvas
    def draw(self):
        # wipe every previously drawn shape off the canvas
        self.canvas.delete("all")
        # local alias for the per-cell pixel size
        size = self.cell_size
        # iterate over every row of the board
        for r in range(self.config.size):
            # iterate over every column of the board
            for c in range(self.config.size):
                # top-left pixel position of this cell
                x1, y1 = c * size, r * size
                # draw one colored square at that position
                rect = self.canvas.create_rectangle(
                    x1, y1, x1 + size, y1 + size,
                    # color comes from the board grid, mapped via PALETTE
                    fill=PALETTE[self.board.grid[r][c]],
                    # dark outline with 2px gap makes cells look separated
                    outline=GRID_BG, width=2,
                )
                # remember the rectangle id so we could update it later
                self.rects[(r, c)] = rect

    # handle a chosen color: apply the flood move and refresh the screen
    def pick(self, color: int):
        # ignore clicks if no board exists or the game has already ended
        if not self.board or self.board.is_over():
            return
        # apply the flood move to the board with the chosen color
        self.board.flood(color)
        # redraw the board to show the newly flooded region
        self.draw()
        # update the move counter / win-lose text
        self.update_status()
        # the board changed, so the best move may have changed too
        self.update_hint()

    # translate a canvas mouse click into a board cell and pick its color
    def on_click(self, event):
        # which column was clicked (pixel x divided by cell width)
        c = event.x // self.cell_size
        # which row was clicked (pixel y divided by cell height)
        r = event.y // self.cell_size
        # local alias for the grid size
        n = self.config.size
        # only react if the click landed on an actual cell
        if 0 <= r < n and 0 <= c < n:
            # flood using the color of the cell that was clicked
            self.pick(self.board.grid[r][c])

    # refresh the hint that shows the greedy policy's recommended next move
    def update_hint(self):
        # no board yet, or the game has ended: there is nothing to recommend
        if self.board is None or self.board.is_over():
            self.hint_label.config(text="Best next move: -", bg=GRID_BG)
            return
        # ask the policy which color it would flood with right now
        color = self.agent.select_move(self.board)
        # a None move is only possible when every option is pointless
        if color is None:
            self.hint_label.config(text="Best next move: -", bg=GRID_BG)
            return
        # paint the hint with the recommended color and its human name
        self.hint_label.config(
            text=f"Best next move: {COLOR_NAMES[color]}",
            bg=PALETTE[color],
        )

    # kick off an animated greedy solve of the current board
    def solve(self):
        # ignore clicks if there is no board or the game has already ended
        if not self.board or self.board.is_over():
            return
        # block re-entry while the animation runs
        self.solve_button.config(state="disabled")
        self.new_button.config(state="disabled")
        self.settings_button.config(state="disabled")
        # start the first solve step
        self._solve_step()

    # performs one greedy move, then schedules the next (animates the solve)
    def _solve_step(self):
        # if the game ended, the solve is complete
        if self.board.is_over():
            self._solve_done()
            return
        # ask the greedy policy for the next move (stuck safety net)
        color = self.agent.select_move(self.board)
        if color is None:
            self._solve_done()
            return
        # apply the chosen move through pick() so drawing/status/hint update
        self.pick(color)
        # schedule the following step unless the game just ended
        if self.board.is_over():
            self._solve_done()
            return
        self.solve_job = self.root.after(
            self.solve_delay, self._solve_step)

    # a solve has finished: re-enable the buttons and clear the job id
    def _solve_done(self):
        # re-enable the buttons that the solver disabled at the start
        self.solve_button.config(state="normal")
        self.new_button.config(state="normal")
        self.settings_button.config(state="normal")
        # nothing to clear if there was no in-flight step to begin with
        if self.solve_job is not None:
            self.solve_job = None
        # refreshes of drawing & status are handled by pick on the last move

    # refresh the status label with moves remaining / result
    def update_status(self):
        # grab a short local reference to the current board
        board = self.board
        # default text shows how many moves the player has left
        status = f"Moves left: {board.moves_left}"
        # if the board is fully one color, show the winning message
        if board.is_solved():
            status = f"Flooded it in {board.moves_used} moves! Well done."
        # otherwise, if no moves remain, show the losing message
        elif board.moves_left == 0:
            status = "Out of moves. Click New Game."
        # push the final text into the label widget
        self.moves_label.config(text=status)


# entry point: parse command-line args, build the window, start the app
def main():
    # create the command-line argument parser with a short description
    parser = argparse.ArgumentParser(description="Flood-It game")
    # register a --size flag, an integer, defaulting to 14
    parser.add_argument("--size", type=int, default=14, help="grid size (default 14)")
    # register a --colors flag, an integer, defaulting to 6
    parser.add_argument("--colors", type=int, default=6, help="number of colors (default 6)")
    # read the arguments that the user actually passed on the command line
    args = parser.parse_args()

    # reject grid sizes outside the playable range (2..26)
    if not (2 <= args.size <= 26):
        parser.error("size must be between 2 and 26")
    # reject color counts that don't fit in the available palette
    if not (2 <= args.colors <= 8):
        parser.error("colors must be between 2 and 8")

    # create the root tkinter window
    root = tk.Tk()
    # instantiate the Game object with the parsed settings (move limit is
    # automatically derived from size and colors by Config)
    Game(root, Config(args.size, args.colors))
    # enter tkinter's event loop, which keeps the window alive until closed
    root.mainloop()


# only run main() when this file is executed directly, not when imported
if __name__ == "__main__":
    main()
