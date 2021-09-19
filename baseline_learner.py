import torch
from torch import nn, optim
import torch.nn.functional as F
import utils
import numpy as np

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



class SampledValueBaseline(BaselineLearner):
    def __init__(self, device, n_samples, *args, **kwargs):
        self.device = torch.device(device)
        self.n_samples = n_samples

    def _get_sampled_qvals(self, q, pi, state, n_samples):
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

    def predict(self, state, q, pi, batch=True):
        state = utils.torch_single_precision(state)
        state = state.to(self.device)
        if not batch:
            state = state.unsqueeze(0)
        
        qvals = self._get_sampled_qvals(q, pi, state, self.n_samples)
        
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


class TDLearner(BaselineLearner):
    def __init__(self, state_dim,
                 discount=0.99, target_update_freq=2,
                 tau=0.005, width=512, depth=2,
                 lr=1e-2, batch_size = 128,
                 device='cpu', load_path = None,
                 **kwargs):
        device = torch.device(device)
        self.device = device
        
        self.net = VMLP(state_dim, width, depth).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        if load_path is not None:
            self.load(load_path)
        self.target_net = VMLP(state_dim, width, depth).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.step_count = 0
        
        self.discount = discount
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.batch_size = batch_size
        self.lr = lr
        self.load_path = load_path
        self.width = width
        self.depth = depth
        self.state_dim = state_dim

    def loss(self, transitions):
        with torch.no_grad():
            sp_value = self.target_net(transitions.sp)
            not_done = 1 - transitions.d
            target_value = transitions.r + not_done * self.discount * sp_value

        pred_value = self.net(transitions.s)
        loss = ((target_value - pred_value) ** 2).mean()

        self.logger.update('baseline/loss', loss.item())
        return loss
    
    def predict(self, state, q=None, pi=None, batch=True):
        state = utils.torch_single_precision(state).to(self.device)
        if not batch:
            state = state.unsqueeze(0)
        value = self.net(state)
        return value if batch else value[0]

    def train_step(self, replay, q=None, pi=None):
        transitions = replay.sample(self.batch_size)
        transitions = transitions.to_device(self.device)
        loss = self.loss(transitions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            utils.soft_update_params(self.net, self.target_net, self.tau)

        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)
        return
    
    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        return

    def eval(self, env, pi, n_episodes, log = True):
        
        returns = np.zeros(n_episodes)
        states = []
        actions = []
        for episode in range(n_episodes):
            state = env.reset()
            action = pi.act(state, batch=False, sample=True)
            states.append(np.expand_dims(state, 0))
            actions.append(np.expand_dims(action, 0))
            ep_return = 0
            done = False
            step = 0
            while not done:
                next_state, reward, done, _ = env.step(action)
                state = next_state
                ep_return += np.power(self.discount, step) * reward
                action = pi.act(state, batch=False, sample=True)
                step += 1
            returns[episode] = ep_return
        states = np.concatenate(states, axis=0)

        preds = self.predict(states).cpu().detach().numpy()
        mse = np.mean((preds - returns)**2)

        if log:
            self.logger.update('baseline/mse', mse)
            self.logger.update('baseline/pred_mean', np.mean(preds))
            self.logger.update('baseline/rollout_mean', np.mean(returns))
        return preds, returns