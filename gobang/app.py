from os.path import join, dirname, abspath
from time import time
from functools import partial
from tkinter import *
from tkinter import messagebox

from gobang.algorithm.ai import *
from gobang.algorithm.policy import CNNPolicy

BOARD_WIDTH = 706
BOARD_SIZE = 15
BOARD_PADDING = 35
CELL_WIDTH = (BOARD_WIDTH - BOARD_PADDING * 2) / (BOARD_SIZE - 1)

LINE_COLOR = '#3d3d3d'
HOME_DIR = dirname(abspath(__file__))
IMAGE_HOME = join(HOME_DIR, "res", "images")

PLAYERS = [
    "Human",
    "Greedy AI",
    "Probabilistic AI",
    "Thinking AI(2s)",
    "Thinking AI(4s)",
    "Thinking AI(8s)",
    "Thinking AI(16s)",
]


class App:

    def __init__(self):
        self.root = Tk()
        self._chess_image_cache = {
            color: PhotoImage(file=join(IMAGE_HOME, f)).subsample(2)
            for color, f in zip([BLACK, WHITE], ["chess_black.gif", "chess_white.gif"])
        }

        self._forbidden_image_cache = None
        self._line_width = 2
        self._box_line_width = 4

        self.human = BLACK
        self.last_chess_hint = None
        self.stones = []
        self.forbidden = []

        self.player = None
        self.player_type = IntVar(value=2)
        self._build_player(2)

        self._build_ui()
        self._build_menu()

        self.state = None
        self.reset_game()

    def run(self):
        self.root.mainloop()

    def invoke_after(self, delay, func):
        self.root.after(delay, func)

    def _build_ui(self):
        self.root.title("Gobang")
        self.root.resizable(False, False)

        frame = Frame(self.root)
        frame.pack()

        #
        self.canvas = Canvas(frame, width=BOARD_WIDTH, height=BOARD_WIDTH)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.pack()

        self._draw_board()
        self._draw_grids()
        self._draw_solids()

    def _build_menu(self):
        # menu
        self.root.option_add('*tearOff', False)
        menu = Menu(self.root)
        self.root.config(menu=menu)

        # create a menu
        event_menu = Menu(menu)
        menu.add_cascade(label="Game", menu=event_menu)
        event_menu.add_command(label="New Game", command=self.reset_game)
        event_menu.add_command(label="Switch Color", command=self.switch_black)

        opponent_menu = Menu(menu)
        menu.add_cascade(label='Opponent', menu=opponent_menu)
        for i, p in enumerate(PLAYERS):
            opponent_menu.add_radiobutton(label=p, value=i, variable=self.player_type,
                                          state=DISABLED if i > 2 else NORMAL,
                                          command=partial(self._build_player, player_type=i))

    def _draw_board(self):
        self.board_image = PhotoImage(file=join(IMAGE_HOME, 'chess_board.gif')).subsample(2)
        self.canvas.create_image(BOARD_WIDTH / 2, BOARD_WIDTH / 2, image=self.board_image)

    def _draw_grids(self):
        """draw board grids
        """
        start, end = BOARD_PADDING, BOARD_WIDTH - BOARD_PADDING
        for row in range(BOARD_SIZE):
            y = row * CELL_WIDTH + BOARD_PADDING
            self.canvas.create_line(start, y, end, y, fill=LINE_COLOR, width=self._line_width)
        for col in range(BOARD_SIZE):
            x = col * CELL_WIDTH + BOARD_PADDING
            self.canvas.create_line(x, start, x, end, fill=LINE_COLOR, width=self._line_width)

        box_padding = 8
        half_width = self._box_line_width / 2
        sp, ep = start - box_padding, end + box_padding
        starts = [(sp - half_width, sp), (sp - half_width, ep), (sp, sp), (ep, sp)]
        ends = [(ep + half_width, sp), (ep + half_width, ep), (sp, ep), (ep, ep)]
        for xp, yp in zip(starts, ends):
            self.canvas.create_line(*xp, *yp, fill=LINE_COLOR, width=self._box_line_width)

    def _draw_circle(self, move, radius, color=LINE_COLOR):
        orig_x, orig_y = self._coords_from_move(move)
        return self.canvas.create_oval(orig_x - radius, orig_y - radius, orig_x + radius, orig_y + radius,
                                       fill=color, width=0)

    def _draw_solids(self):
        radius = min(CELL_WIDTH / 5, 5)
        indices = [3, 7, 11]
        for x in indices:
            for y in indices:
                if x == y or x + y == 14:
                    self._draw_circle((x, y), radius)

    def _draw_chess(self, move, color):
        image = self._chess_image_cache[color]
        return self.canvas.create_image(self._coords_from_move(move), image=image)

    def _draw_forbidden(self):
        if self._forbidden_image_cache is None:
            image = PhotoImage(file=join(IMAGE_HOME, "chess_forbidden.gif")).subsample(2)
            self._forbidden_image_cache = image

        for fb in self.forbidden:
            self.canvas.delete(fb)
        self.forbidden = []

        for x in range(self.state.size):
            for y in range(self.state.size):
                move = (x, y)
                if self.state.board[x][y] == EMPTY and not self.state.is_legal(move):
                    self.forbidden.append(
                        self.canvas.create_image(self._coords_from_move(move), image=self._forbidden_image_cache))

    @staticmethod
    def _coords_from_move(move):
        x, y = move
        return BOARD_PADDING + CELL_WIDTH * x, BOARD_PADDING + CELL_WIDTH * y

    @staticmethod
    def _move_from_coords(x, y):
        move = tuple(int((x - BOARD_PADDING) / CELL_WIDTH + 0.5) for x in [x, y])
        for index in move:
            if index < 0 or index >= BOARD_SIZE:
                return None
        return move

    def human_play(self, move):
        if self.state.is_legal(move):
            self._do_play_chess(move)

            if self.player:
                self.invoke_after(10, self.play_chess_if_needed)

    def play_chess_if_needed(self):
        if not self.state.is_end_of_game and not self.is_human_turn():
            now = time()
            move = self.player.get_move(self.state)
            self._do_play_chess(move)
            print("time cost:{:.3f}, {}".format(time() - now, move))

    def _do_play_chess(self, move):
        if self.state.is_end_of_game:
            return

        color = self.state.current_player
        self.state.do_move(move, verbose=False)

        chess = self._draw_chess(move, color)
        self.stones.append(chess)

        self._update_last_hint(move)

        self._draw_forbidden()

        if self.state.is_end_of_game:
            msgs = {BLACK: "Black Win", WHITE: "White Win", EMPTY: "Draw"}
            messagebox.showinfo("Game Over", msgs[self.state.winner])

        # print('move: ', move);
        # self.state.print_board()

    def _delete_last_hint(self):
        if self.last_chess_hint is not None:
            self.canvas.delete(self.last_chess_hint)
            self.last_chess_hint = None

    def _update_last_hint(self, move):
        self._delete_last_hint()

        self.last_chess_hint = self._draw_circle(move, 5, color='red')

    def is_human_turn(self):
        return self.state.current_player == self.human

    def reset_game(self):
        self.state = GameState()
        self._draw_forbidden()

        for chess in self.stones:
            self.canvas.delete(chess)
        self._delete_last_hint()

        self.play_chess_if_needed()

    def switch_black(self):
        self.human = -self.human
        self.play_chess_if_needed()

    def _build_player(self, player_type):
        # "Human",
        # "Greedy AI",
        # "Probabilistic AI",
        # "Thinking AI(2s)",
        # "Thinking AI(4s)",
        # "Thinking AI(8s)",
        # "Thinking AI(16s)",
        self.policy = CNNPolicy.load_model(join(HOME_DIR, "data", "model.pth"))
        if player_type == 0:
            self.player = None
        if player_type == 1:
            self.player = GreedyPolicyPlayer(self.policy)
        elif player_type == 2:
            self.player = ProbabilisticPolicyPlayer(self.policy, temperature=0.1)
        # else:
        #     self.value = CNNValue.load_model('models/value_model_47p_3331.json')
        #     self.value.model.load_weights('value_rl_47/weights.00000.hdf5')
        #
        #     self.player = MCTSPlayer(self.value.eval_state, self.policy.eval_state,
        #                              None, lmbda=0, playout_depth=5, rollout_limit=0, n_playout=10,
        #                              time_limit=2**(player_type - 2))

    def on_click(self, event):
        move = self._move_from_coords(event.x, event.y)
        if move is None:
            return
        self.human_play(move)


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
