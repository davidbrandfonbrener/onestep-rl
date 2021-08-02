import torch
from torch import nn, optim
import torch.nn.functional as F
import utils

class BaselineLearner(object):
    def __init__(self, *args, **kwargs):
        return
    
    def set_logger(self, logger):
        self.logger = logger
    
    def loss(self, transitions, q, pi):
        raise NotImplementedError

    def predict(self, state, q, pi, batch=True):
        raise NotImplementedError
    
    def train_step(self, replay, q, pi):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError


def _get_sampled_qvals(q, pi, state, n_samples):
    batch_size = state.shape[0]
    state_shape = tuple(state.shape[1:])

    # sample actions
    dist = pi.pinet(state)
    actions = dist.sample((n_samples,))
    
    # reshape
    action_shape = tuple(actions.shape[2:])
    actions = actions.reshape((n_samples * batch_size, *action_shape))
    
    # calculate q values
    ones = tuple([1 for i in range(len(state.shape))])
    repeated_states = state.unsqueeze(0).repeat((n_samples, *ones))
    repeated_states = repeated_states.reshape((n_samples * 
                                            batch_size, *state_shape))
    qvals = q.predict(repeated_states, actions)
    qvals = qvals.reshape((n_samples, batch_size, 1))

    return qvals


class SampledValueBaseline(BaselineLearner):
    def __init__(self, device, n_samples, *args, **kwargs):
        self.device = torch.device(device)
        self.n_samples = n_samples

    def predict(self, state, q, pi, batch=True):
        state = utils.torch_single_precision(state)
        state = state.to(self.device)
        if not batch:
            state = state.unsqueeze(0)
        
        qvals = _get_sampled_qvals(q, pi, state, self.n_samples)
        
        # compute mean
        value = qvals.mean(dim=0)
        
        return value if batch else value[0]


class VMLP(nn.Module):
    def __init__(self, state_dim, width, depth):
        super().__init__()
        self.net = utils.MLP(input_shape=(state_dim), output_dim=1,
                        width=width, depth=depth)

    def forward(self, s):
        x = torch.flatten(s, start_dim=1)
        return self.net(x)