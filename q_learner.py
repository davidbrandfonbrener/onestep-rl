import torch
from torch import optim
import numpy as np
import utils
import q_network

class QLearner(object):
    def __init__(self, state_dim, action_dim,
                 discount=0.99, 
                 target_update_freq=2,
                 tau=0.005,
                 width=512,
                 depth=2,
                 lr=1e-2,
                 batch_size = 128,
                 device='cpu',
                 load_path = None,
                 **kwargs):
        device = torch.device(device)
        self.device = device
        
        self.qnet = q_network.QMLP(state_dim, action_dim, 
                                    width, depth).to(device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        if load_path is not None:
            self.load(load_path)
        self.target_qnet = q_network.QMLP(state_dim, action_dim, 
                                    width, depth).to(device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
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
        self.action_dim = action_dim

    def set_logger(self, logger):
        self.logger = logger

    def loss(self, transitions, pi, beta=None):
        raise NotImplementedError

    def train_step(self, replay, pi, beta=None):
        transitions = replay.sample(self.batch_size)
        transitions = transitions.to_device(self.device)
        loss = self.loss(transitions, pi, beta)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            utils.soft_update_params(self.qnet, self.target_qnet, self.tau)

        return loss.item()

    def predict(self, state, action, batch=True):
        state = utils.torch_single_precision(state)
        action = utils.torch_single_precision(action)
        state = state.to(self.device)
        action = action.to(self.device)
        if not batch:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
        value = self.qnet(state, action)
        return value if batch else value[0]

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
        actions = np.concatenate(actions, axis=0)

        preds = self.predict(states, actions).cpu().detach().numpy()
        mse = np.mean((preds - returns)**2)

        if log:
            self.logger.update('q/mse', mse)
            self.logger.update('q/pred_mean', np.mean(preds))
            self.logger.update('q/rollout_mean', np.mean(returns))
        return preds, returns

    def save(self, path):
        torch.save(self.qnet.state_dict(), path)
        return
    
    def load(self, path):
        self.qnet.load_state_dict(torch.load(path, map_location=self.device))
        return

class BanditQLearner(QLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, transitions, pi=None, beta=None):
        pred_value = self.qnet(transitions.s, transitions.a)
        loss = ((transitions.r - pred_value) ** 2).mean()

        self.logger.update('q/loss', loss.item())
        return loss

class SarsaLearner(QLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, transitions, pi=None, beta=None):
        with torch.no_grad():
            sp_value = self.target_qnet(transitions.sp, transitions.ap)
            not_done = 1 - transitions.d
            target_value = transitions.r + not_done * self.discount * sp_value

        pred_value = self.qnet(transitions.s, transitions.a)
        loss = ((target_value - pred_value) ** 2).mean()

        self.logger.update('q/loss', loss.item())
        return loss

class QPiLearner(QLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, transitions, pi, beta=None):
        """Uses the policy to get a better estimate of the value of
        the next state. Works with off-policy data."""
        with torch.no_grad():
            ap = pi.act(transitions.sp, batch=True, sample=True)
            sp_value = self.target_qnet(transitions.sp, ap)
            not_done = 1 - transitions.d
            target_value = transitions.r + not_done * self.discount * sp_value  

        pred_value = self.qnet(transitions.s, transitions.a)
        loss = ((target_value - pred_value) ** 2).mean()

        self.logger.update('q/loss', loss.item())
        return loss

class DoubleQLearner(QLearner):
    def __init__(self, state_dim, action_dim, *args, **kwargs):
        super().__init__(state_dim, action_dim, *args, **kwargs)
        self.qnet = q_network.DoubleQMLP(state_dim, action_dim, 
                                    self.width, self.depth).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        if self.load_path is not None:
            self.qnet.load_state_dict(torch.load(self.load_path, 
                                        map_location=self.device))
        self.target_qnet = q_network.DoubleQMLP(state_dim, action_dim, 
                                    self.width, self.depth).to(self.device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def minQ(self, state, action):
        Q1, Q2 = self.target_qnet(state, action)
        return torch.min(Q1, Q2)

    def target_minQ(self, state, action):
        Q1, Q2 = self.qnet(state, action)
        return torch.min(Q1, Q2)

    def predict(self, state, action, batch=True):
        state = utils.torch_single_precision(state)
        action = utils.torch_single_precision(action)
        state = state.to(self.device)
        action = action.to(self.device)
        if not batch:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
        value = self.minQ(state, action)
        return value if batch else value[0]

class DoubleQPiLearner(DoubleQLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, transitions, pi):
        with torch.no_grad():
            ap = pi.act(transitions.sp)
            not_done = 1 - transitions.d
            target_value = transitions.r + \
                        not_done * self.discount * self.target_minQ(transitions.sp, ap)

        pred1, pred2 = self.qnet(transitions.s, transitions.a)
        loss = ((target_value - pred1) ** 2 + 
                (target_value - pred2) ** 2).mean()

        self.logger.update('q/loss', loss.item())
        return loss

class DoubleSarsaLearner(DoubleQLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, transitions, pi):
        with torch.no_grad():
            not_done = 1 - transitions.d
            target_value = transitions.r + \
                        not_done * self.discount * self.target_minQ(transitions.sp, transitions.ap)

        pred1, pred2 = self.qnet(transitions.s, transitions.a)
        loss = ((target_value - pred1) ** 2 + 
                (target_value - pred2) ** 2).mean()

        self.logger.update('q/loss', loss.item())
        return loss




