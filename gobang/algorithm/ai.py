"""Policy players"""
from gobang.game import *
from . import mcts
from operator import itemgetter


class GreedyPolicyPlayer(object):
    """A player that uses a greedy policy (i.e. chooses the highest probability
    move each turn)
    """

    def __init__(self, policy_function, move_limit=None):
        self.policy = policy_function
        self.move_limit = move_limit

    def get_move(self, state):
        if self.move_limit is not None and len(state.history) > self.move_limit:
            return PASS_MOVE
        sensible_moves = [move for move in state.get_legal_moves()]
        if len(sensible_moves) > 0:
            move_probs = self.policy.eval_state(state, sensible_moves)
            assert len(move_probs) > 0
            max_prob = max(move_probs, key=itemgetter(1))
            return max_prob[0]
        # No 'sensible' moves available, so do pass move
        return PASS_MOVE

    def get_moves(self, states):
        """Batch version of get_move. A list of moves is returned (one per state)
        """
        sensible_move_lists = [[move for move in st.get_legal_moves()]
                               for st in states]
        all_moves_distributions = self.policy.batch_eval_state(states, sensible_move_lists)
        move_list = [None] * len(states)
        for i, move_probs in enumerate(all_moves_distributions):
            max_prob = max(move_probs, key=itemgetter(1))
            move_list[i] = max_prob[0]
        return move_list

    def do_move(self, move):
        pass


class ProbabilisticPolicyPlayer(object):
    """A player that samples a move in proportion to the probability given by the
    policy.

    By manipulating the 'temperature', moves can be pushed towards totally random
    (high temperature) or towards greedy play (low temperature)
    """

    def __init__(self, policy_function, temperature=1.0, move_limit=None, greedy_start=None):
        assert (temperature > 0.0)
        self.policy = policy_function
        self.move_limit = move_limit
        self.beta = 1.0 / temperature
        self.greedy_start = greedy_start

    def apply_temperature(self, distribution):
        log_probabilities = np.log(distribution)
        # apply beta exponent to probabilities (in log space)
        log_probabilities *= self.beta
        # scale probabilities to a more numerically stable range (in log space)
        log_probabilities = log_probabilities - log_probabilities.max()
        # convert back from log space
        probabilities = np.exp(log_probabilities)
        # re-normalize the distribution
        return probabilities / probabilities.sum()

    def get_move(self, state):
        if self.move_limit is not None and len(state.history) > self.move_limit:
            return PASS_MOVE
        sensible_moves = [move for move in state.get_legal_moves()]
        if len(sensible_moves) > 0:
            move_probs = self.policy.eval_state(state, sensible_moves)
            if self.greedy_start is not None and len(state.history) >= self.greedy_start:
                # greedy
                max_prob = max(move_probs, key=itemgetter(1))
                return max_prob[0]
            else:
                # zip(*list) is like the 'transpose' of zip;
                # zip(*zip([1,2,3], [4,5,6])) is [(1,2,3), (4,5,6)]
                # print(move_probs)
                moves, probabilities = zip(*move_probs)
                # apply 'temperature' to the distribution
                probabilities = self.apply_temperature(probabilities)
                # numpy interprets a list of tuples as 2D, so we must choose an
                # _index_ of moves then apply it in 2 steps
                choice_idx = np.random.choice(len(moves), p=probabilities)
                return moves[choice_idx]
        return PASS_MOVE

    def get_moves(self, states):
        """Batch version of get_move. A list of moves is returned (one per state)
        """
        sensible_move_lists = [[move for move in st.get_legal_moves()]
                               for st in states]
        all_moves_distributions = self.policy.batch_eval_state(states, sensible_move_lists)
        move_list = [None] * len(states)
        for i, move_probs in enumerate(all_moves_distributions):
            if len(move_probs) == 0 or self.move_limit is not None and len(states[i].history) > self.move_limit:
                move_list[i] = PASS_MOVE
            else:
                if self.greedy_start is not None and len(states[i].history) >= self.greedy_start:
                    # greedy
                    max_prob = max(move_probs, key=itemgetter(1))
                    move_list[i] = max_prob[0]
                else:
                    # probabilistic
                    moves, probabilities = zip(*move_probs)
                    probabilities = np.array(probabilities)
                    probabilities **= self.beta
                    probabilities = probabilities / probabilities.sum()
                    choice_idx = np.random.choice(len(moves), p=probabilities)
                    move_list[i] = moves[choice_idx]
        return move_list

    def do_move(self, move):
        pass


class MCTSPlayer(object):
    def __init__(self, value_function, policy_function, rollout_function, lmbda=.5, c_puct=5,
                 rollout_limit=20, playout_depth=10, n_playout=100):
        self.mcts = mcts.MCTS(value_function, policy_function, rollout_function, lmbda, c_puct,
                              rollout_limit, playout_depth, n_playout)

    def get_move(self, state):
        sensible_moves = [move for move in state.get_legal_moves()]
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(state)
            self.do_move(move)
            return move
        # No 'sensible' moves available, so do pass move
        return PASS_MOVE

    def do_move(self, move):
        self.mcts.update_with_move(move)
