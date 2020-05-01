import numpy as np

WHITE = -1
BLACK = +1
EMPTY = 0
SIZE = 15
TOTAL_CHESS = SIZE * SIZE

PASS_MOVE = None

# The forbidden hand
# 三三禁手: 黑棋一子落下同时形成两个活三
# 四四禁手: 黑棋一子落下同时形成两个或两个以上的冲四或活四
# 长连禁手: 黑棋一子落下形成一个或一个以上的长连

PATTERN_NONE = 0
PATTERN_LIVE3 = 1  # 活三：本方再走一着可以形成活四的三
PATTERN_RUSH4 = 2  # 冲四：只有一个点可以成五的四
PATTERN_LIVE4 = 3  # 活四：有两个点可以成五的四
PATTERN_LINK5 = 4  # 五连
PATTERN_LINK6 = 5  # 长连：形成的5个以上同色棋子不间隔的相连


class IllegalMove(Exception):
    pass


class GameState(object):
    """the game state object

    :param size: the size for the game board
    :param board: 2D numpy array
    :param black_banned: use the black forbidden hand
    :param start_banned: use the starting forbidden hand
    """

    def __init__(self, size=SIZE, board=None, black_banned=True, start_banned=True):
        self.size = size

        if board is None:
            self.board = np.zeros((size, size), dtype=int)
            self.board.fill(EMPTY)
        else:
            self.board = board

        steps = (self.board != 0).sum()
        if steps > 0:
            self._update_types_for_all()
        self.__current_player = BLACK if steps % 2 == 0 else WHITE

        self.history = [(0, 0)] * steps
        self.is_end_of_game = False
        self.winner = 0

        # cache of list of legal moves (actually 'sensible' moves, with a
        # separate list for eye-moves on request)
        self.__legal_move_cache = None
        # on-the-fly record of 'age' of each stone
        self.stone_ages = np.zeros((size, size), dtype=np.int) - 1

        # forbidden hands
        self.black_banned = black_banned
        self.start_banned = start_banned

        # 2 colors, 4 directions
        self.blank_types = np.zeros((size, size, 2, 4), dtype=int)
        self.black_banes = np.full((size, size), False)

    def print_board(self):
        cs = {-1: "-", 1: "+", 0: " "}
        for y in range(self.size):
            for x in range(self.size):
                print(cs[self.board[x][y]], end=" ")
            print(' ')

    def _on_board(self, position):
        """simply return True iff position is within the bounds of [0, self.size)
        """
        (x, y) = position
        return 0 <= x < self.size and 0 <= y < self.size

    def copy(self):
        """get a copy of this Game state
        """
        other = GameState(self.size)
        other.board = self.board.copy()
        other.__current_player = self.__current_player
        other.history = list(self.history)
        other.black_banned = self.black_banned

        return other

    @staticmethod
    def _pattern(stones6, color):
        assert len(stones6) == 6

        color_count = stones6.count(color)
        r_color = -color
        reverse_color_count = stones6.count(r_color)
        if color_count < 3:
            return PATTERN_NONE
        elif color_count > 5:
            return PATTERN_LINK6
        elif color_count == 5:
            return PATTERN_LINK5 if stones6[0] != color or stones6[-1] != color else PATTERN_NONE
        elif color_count == 3:
            return PATTERN_LIVE3 if stones6[0] == 0 and stones6[-1] == 0 and reverse_color_count == 0 else PATTERN_NONE
        elif color_count == 4:
            if stones6[0] == 0 and stones6[-1] == 0:
                return PATTERN_LIVE4
            else:
                if reverse_color_count > 1:
                    return PATTERN_NONE
                elif reverse_color_count == 1:
                    return PATTERN_RUSH4 if stones6[0] == r_color or stones6[-1] == r_color else PATTERN_NONE
                else:
                    return PATTERN_RUSH4 if stones6[0] == 0 or stones6[-1] == 0 else PATTERN_NONE

    def _stones6_from_board(self, action, color=None):
        """iterate over four direction, yield 6 continuous stones each time

        :param action: (x, y)
        :param color: BLACK, WHITE
        :return: (direction, stones)
        """
        ox, oy = action
        for direction, (dx, dy) in enumerate([(1, 0), (0, 1), (1, 1), (1, -1)]):  # 4 directions
            for i in range(5, -1, -1):
                x, y = ox - dx * i, oy - dy * i
                if not self._on_board((x, y)) or not self._on_board((x + dx * 5, y + dy * 5)):
                    continue
                stones6 = [self.board[x + dx * j][y + dy * j] for j in range(6)]
                if color is not None:
                    stones6[i - 6] = color
                yield direction, stones6

    def is_positional_black_banned(self, action):
        """Find all actions that the current_player has done in the past, taking into
        account the fact that history starts with BLACK when there are no
        handicaps or with WHITE when there are.

        """
        return self.__current_player != WHITE and self.black_banes[action]

    def _update_banes_for_pos(self, pos):
        types = self.blank_types[pos][self.type_idx_for_color(BLACK)]
        if (types == PATTERN_LINK5).any():
            self.black_banes[pos] = 0
            return

        c3 = (types == PATTERN_LIVE3).sum()
        c4 = (types == PATTERN_RUSH4).sum() + (types == PATTERN_LIVE4).sum()
        c6 = (types == PATTERN_LINK6).sum()

        self.black_banes[pos] = c3 > 1 or c4 > 1 or c6 > 0

    @staticmethod
    def type_idx_for_color(color):
        return 0 if color == BLACK else 1

    def _update_types_for_pos(self, pos):
        for color in [BLACK, WHITE]:
            types = self.blank_types[pos][self.type_idx_for_color(color)]
            types.fill(0)

            for direction, stones6 in self._stones6_from_board(pos, color=color):
                types[direction] = max(types[direction], self._pattern(stones6, color))

        self._update_banes_for_pos(pos)

    def _update_types_for_move(self, action):
        """invoked when the do_move method is invoked"""
        ox, oy = action
        self.blank_types[action].fill(PATTERN_NONE)

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for idx, (dx, dy) in enumerate(directions):
            for i in range(-5, 6):
                pos = ox - dx * i, oy - dy * i
                if not self._on_board(pos) or self.board[pos] != EMPTY:
                    continue
                self._update_types_for_pos(pos)

    def _update_types_for_all(self):
        """invoked when reset all board"""
        self.blank_types.fill(PATTERN_NONE)

        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == EMPTY:
                    self._update_types_for_pos((x, y))

    def _is_init_rules_accept(self, action):
        if not self.start_banned:
            return True

        steps = len(self.history)
        if steps > 2:
            return True

        center = self.size // 2
        if steps == 0:
            return action == (center, center)

        dx, dy = abs(action[0] - center), abs(action[1] - center)
        if steps == 1:
            return dx <= 1 and dy <= 1
        else:
            return dx <= 2 and dy <= 2

    def is_legal(self, action):
        """determine if the given action (x,y) is a legal move
        """
        if not self._on_board(action):
            return False
        if self.board[action] != EMPTY:
            return False
        if self.start_banned and not self._is_init_rules_accept(action):
            return False
        if self.black_banned and self.is_positional_black_banned(action):
            return False
        return True

    def get_legal_moves(self):
        if self.__legal_move_cache is not None:
            return self.__legal_move_cache

        self.__legal_move_cache = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_legal((x, y)):
                    self.__legal_move_cache.append((x, y))
        return self.__legal_move_cache

    def _check_is_game_over(self, action, verbose):
        if len(self.history) == TOTAL_CHESS:
            self.is_end_of_game = True
            if verbose:
                print("Draw")
            return True

        color = self.board[action]

        for direction, stones6 in self._stones6_from_board(action, color=None):
            if self._pattern(stones6, color) == PATTERN_LINK5:
                self.is_end_of_game = True
                self.winner = color
                if verbose:
                    print("Win")
                return True

    def get_winner(self):
        return self.winner

    @property
    def current_player(self):
        """Returns the color of the player who will make the next move.
        """
        return self.__current_player

    def do_move(self, action, verbose=False):
        """Play stone at action=(x,y).
        If it is a legal move, current_player switches to the opposite color
        If not, an IllegalMove exception is raised
        """
        if action is PASS_MOVE:
            self.__current_player = -self.current_player
        elif self.is_legal(action):
            # increment age of stones by 1
            self.stone_ages[self.stone_ages >= 0] += 1

            # new stone
            color = self.__current_player
            (x, y) = action
            self.board[x][y] = color
            self.stone_ages[x][y] = 0

            # next turn
            self.__current_player = -color
            self.history.append(action)
            self.__legal_move_cache = None
        else:
            raise IllegalMove(str(action))

        # update types for blanks
        self._update_types_for_move(action)

        # Check for end of game
        if len(self.history) >= 9:
            self._check_is_game_over(action, verbose)
        return self.is_end_of_game
