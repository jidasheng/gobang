from gobang.game import *


def get_board(state):
    """A feature encoding WHITE BLACK and EMPTY on separate planes, but plane 0
    always refers to the current player and plane 1 to the opponent
    """
    planes = np.zeros((state.size, state.size, 3))
    planes[:, :, 0] = state.board == state.current_player  # own stone
    planes[:, :, 1] = state.board == -state.current_player  # opponent stone
    planes[:, :, 2] = state.board == EMPTY  # empty space
    return planes


def get_blank_types(state, maximum=5):
    """
    """
    nums_line = 4
    nums_color = 2
    planes = np.zeros((state.size, state.size, maximum * nums_line * nums_color))

    current_color = state.current_player
    for idx, color in enumerate([current_color, -current_color]):
        color_idx = state.type_idx_for_color(color)
        for x in range(state.size):
            for y in range(state.size):
                for line in range(nums_line):
                    blank_type = state.blank_types[x][y][color_idx][line]
                    assert 0 <= blank_type <= maximum
                    if blank_type > 0:
                        planes[x, y, (idx * nums_line + line) * maximum + blank_type - 1] = 1
    return planes


def get_black_banes(state):
    planes = np.zeros((state.size, state.size, 2))
    if state.black_banned:
        ban_idx = 0 if state.current_player == BLACK else 1
        planes[state.black_banes, ban_idx] = 1
    return planes


def get_turns_since(state, maximum=8):
    """A feature encoding the age of the stone at each location up to 'maximum'

    Note:
    - the [maximum-1] plane is used for any stone with age greater than or equal to maximum
    - EMPTY locations are all-zero features
    """
    planes = np.zeros((state.size, state.size, maximum))
    for x in range(state.size):
        for y in range(state.size):
            if state.stone_ages[x][y] >= 0:
                planes[x, y, min(state.stone_ages[x][y], maximum - 1)] = 1
    return planes


def get_legal(state):
    """Zero at all illegal moves, one at all legal moves. Unlike sensibleness, no eye check is done
    """
    feature = np.zeros((state.size, state.size, 1))
    for (x, y) in state.get_legal_moves():
        feature[x, y, 0] = 1
    return feature


# named features and their sizes are defined here
FEATURES = {
    "board": {
        "size": 3,
        "function": get_board
    },
    "ones": {
        "size": 1,
        "function": lambda state: np.ones((state.size, state.size, 1))
    },
    "blank_types": {
        "size": 40,
        "function": get_blank_types
    },
    "black_ban": {
        "size": 2,
        "function": get_black_banes
    },
    "turns_since": {
        "size": 8,
        "function": get_turns_since
    },
    "zeros": {
        "size": 1,
        "function": lambda state: np.zeros((state.size, state.size, 1))
    },
    "legal": {
        "size": 1,
        "function": get_legal
    }
}

DEFAULT_FEATURES = [
    "board", "ones", "blank_types", "black_ban", "zeros"
]


class Preprocess(object):
    """a class to convert from AlphaGo GameState objects to tensors of one-hot
    features for NN inputs
    """

    def __init__(self, feature_list=DEFAULT_FEATURES):
        """create a preprocessor object that will concatenate together the
        given list of features
        """

        self.output_dim = 0
        self.feature_list = feature_list
        self.processors = []
        for i, feat in enumerate(feature_list):
            feat_ = feat.lower()
            if feat_ in FEATURES:
                self.processors.append(FEATURES[feat_]["function"])
                self.output_dim += FEATURES[feat_]["size"]
            else:
                raise ValueError("unknown feature: %s" % feat)

    def state_to_tensor(self, state):
        feat_list = [proc(state) for proc in self.processors]
        # concatenate along feature dimension then add in a singleton 'batch' dimension
        # f, s = self.output_dim, state.size
        feat_tensors = np.concatenate(feat_list, axis=-1)
        return feat_tensors.transpose((2, 0, 1))[np.newaxis, :]
