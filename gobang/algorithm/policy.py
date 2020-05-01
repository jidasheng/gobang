import torch
import torch.nn as nn
import numpy as np
from gobang.preprocessing import Preprocess


def flatten_idx(position, size):
    x, y = position
    return x * size + y


def unflatten_idx(idx, size):
    x, y = divmod(idx, size)
    return x, y


class CNNPolicy(nn.Module):
    """uses a CNN to evaluate the state of the game
    and compute a probability distribution over the next action
    """

    def __init__(self, preprocessor, hidden_size=32):
        super(CNNPolicy, self).__init__()

        self.preprocessor = preprocessor
        self.cnn1 = nn.Conv2d(preprocessor.output_dim, hidden_size, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.cnn4 = nn.Conv2d(hidden_size, 1, kernel_size=1)

    def forward(self, features):
        outputs = torch.relu(self.cnn1(features))
        outputs = torch.relu(self.cnn2(outputs))
        outputs = torch.relu(self.cnn3(outputs))
        outputs = self.cnn4(outputs).view(features.size(0), -1)
        outputs = torch.softmax(outputs, dim=-1)
        return outputs

    @staticmethod
    def load_model(pth_file, device=None):
        policy = CNNPolicy(Preprocess())

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        policy.device = device

        state_dict = torch.load(pth_file, map_location=device)
        policy.load_state_dict(state_dict)
        policy.to(device)
        return policy

    @staticmethod
    def _select_moves_and_normalize(nn_output, moves, size):
        """helper function to normalize a distribution over the given list of moves
        and return a list of (move, prob) tuples
        """
        if len(moves) == 0:
            return []
        move_indices = [flatten_idx(m, size) for m in moves]
        # get network activations at legal move locations
        distribution = nn_output[move_indices]
        total = distribution.sum()
        if total == 0:
            print('*** invalid inference')
            distribution.fill(1 / len(distribution))
        else:
            distribution = distribution / total
        return list(zip(moves, distribution))

    def batch_eval_state(self, states, moves_lists=None):
        """Given a list of states, evaluates them all at once to make best use of GPU
        batching capabilities.

        Analogous to [eval_state(s) for s in states]

        Returns: a parallel list of move distributions as in eval_state
        """
        n_states = len(states)
        if n_states == 0:
            return []
        state_size = states[0].size
        if not all([st.size == state_size for st in states]):
            raise ValueError("all states must have the same size")
        # concatenate together all one-hot encoded states along the 'batch' dimension
        nn_input = np.concatenate([self.preprocessor.state_to_tensor(s) for s in states], axis=0)
        # pass all input through the network at once (backend makes use of
        # batches if len(states) is large)
        network_output = self.forward(nn_input)
        # default move lists to all legal moves
        moves_lists = moves_lists or [st.get_legal_moves() for st in states]
        results = [
            self._select_moves_and_normalize(network_output[i], moves_lists[i], state_size)
            for i in range(n_states)
        ]
        return results

    def eval_state(self, state, moves=None):
        if state.is_end_of_game:
            return []

        with torch.no_grad():
            tensor = torch.from_numpy(self.preprocessor.state_to_tensor(state)).float().to(self.device)
            # run the tensor through the network
            network_output = self(tensor)
            # print(network_output)
            moves = moves or state.get_legal_moves()
            return self._select_moves_and_normalize(network_output[0].cpu().numpy(), moves, state.size)
