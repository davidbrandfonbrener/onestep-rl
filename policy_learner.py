import torch
from torch import optim
import numpy as np
import utils
import policy_network
import q_network
from copy import deepcopy

def log_prob_func(dist, sample):
    log_prob = dist.log_prob(sample)
    if len(log_prob.shape) == 1:
        return log_prob
    else:
        return log_prob.sum(-1, keepdim=True)

def stable_action(action):
    epsilon = 1e-4
    stable = action.clamp(-1. + epsilon, 1. - epsilon)
    return stable

def soft_clamp(x, low, high):
    x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x

class PiLearner(object):
    def __init__(self, state_dim, action_dim,
                 logger_type = 'pi', # options are 'pi' or 'beta'
                 dist_type='trunc', # options are 'normal', 'trunc', or 'squash'
                 width=1024,
                 depth=2,
                 lr=1e-4,
                 batch_size = 512,
                 device='cpu',
                 load_path = None,
                 **kwargs):
        device = torch.device(device)
        self.device = device
        self.dist_type = dist_type
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.lr = lr
        self.logger_type = logger_type

        self.pinet = policy_network.GaussMLP(state_dim, action_dim, 
                                            width, depth, dist_type).to(device)
        if load_path is not None:
            self.load(load_path)
        self.optimizer = optim.Adam(self.pinet.parameters(), lr=lr)

    def set_logger(self, logger):
        self.logger = logger

    def loss(self, transitions, q, baseline, beta):
        raise NotImplementedError

    def train_step(self, replay, q, baseline, beta):
        transitions = replay.sample(self.batch_size)
        transitions = transitions.to_device(self.device)
        loss = self.loss(transitions, q, baseline, beta)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def act(self, state, batch=True, sample=True):
        state = utils.torch_single_precision(state)
        state = state.to(self.device)
        if not batch:
            state = state.unsqueeze(0)
        dist = self.pinet(state)
        if sample:
            action = dist.sample()
        elif self.dist_type == 'mix':
            action = utils.mode(dist)
        else:    
            action = dist.mean
        # clip 
        action = action.clamp(-1., 1.)
        return action if batch else action[0].cpu().detach().numpy()

    def eval(self, env, n_episodes):
        returns = np.zeros(n_episodes)
        for episode in range(n_episodes):
            done = False
            state = env.reset()
            step, ep_return = 0, 0
            while not done:
                action = self.act(state, batch=False, sample=False)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                ep_return += reward
                step += 1
            returns[episode] = ep_return
        ret = np.mean(returns)
        
        self.logger.update(self.logger_type + '/return', ret)
        return ret

    def save(self, path):
        torch.save(self.pinet.state_dict(), path)
        return
    
    def load(self, path):
        self.pinet.load_state_dict(torch.load(path, map_location=self.device))
        return

    def load_from_pilearner(self, learner):
        self.pinet = deepcopy(learner.pinet)
        self.optimizer = optim.Adam(self.pinet.parameters(), lr=self.lr)



class BCLearner(PiLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def loss(self, transitions, q, baseline=None, beta=None):
        dist = self.pinet(transitions.s)
        log_prob = log_prob_func(dist, transitions.a) 
        loss = (-log_prob).mean()

        with torch.no_grad():
            self.logger.update(self.logger_type + '/loss', loss.item())
            action = dist.rsample()
            log_prob = log_prob_func(dist, action) 
            self.logger.update(self.logger_type + '/entropy', 
                                -log_prob.mean().item())
            if q is not None:
                qval = q.predict(transitions.s, action)
                self.logger.update(self.logger_type + '/qval', 
                                    qval.mean().item())
        return loss

class GreedyLearner(PiLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def loss(self, transitions, q, baseline=None, beta=None):
        dist = self.pinet(transitions.s)
        action = dist.rsample()
        qval = q.predict(transitions.s, action)
        loss = (- qval).mean()

        self.logger.update('pi/loss', loss.item())
        log_prob = log_prob_func(dist, action) 
        self.logger.update('pi/entropy', -log_prob.mean().item())
        self.logger.update('pi/qval', qval.mean().item())
        return loss

class ReverseKLRegLearner(PiLearner):
    def __init__(self, alpha, n_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.n_samples = n_samples

    def loss(self, transitions, q, baseline, beta):
        dist = self.pinet(transitions.s)
        action = dist.rsample()
        qval = q.predict(transitions.s, action)

        kl_actions = dist.rsample(sample_shape=(self.n_samples,))
        pi_log_prob = log_prob_func(dist, kl_actions)
        beta_dist = beta.pinet(transitions.s)
        beta_log_prob = log_prob_func(beta_dist, kl_actions)
        kl = (pi_log_prob - beta_log_prob).mean(dim=0)
       
        loss = (- qval + self.alpha * kl).mean()
        
        self.logger.update('pi/loss', loss.item())
        self.logger.update('pi/entropy', -pi_log_prob.mean().item())
        self.logger.update('pi/kl', kl.mean().item())
        self.logger.update('pi/qval', qval.mean().item())
        return loss

class ExpWeightLearner(PiLearner):
    def __init__(self, temp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = temp
        self.max_weight = 100.0

    def loss(self, transitions, q, baseline, beta):
        qval = q.predict(transitions.s, transitions.a)
        b = baseline.predict(transitions.s, q, beta)

        dist = self.pinet(transitions.s)
        IL_log_prob = log_prob_func(dist, stable_action(transitions.a))  
        
        weight = (self.temp * (qval - b)).exp().clamp(0, self.max_weight)
        loss = (- weight * IL_log_prob).mean()

        self.logger.update('pi/loss', loss.item())
        action = dist.sample()
        log_prob = log_prob_func(dist, action) 
        self.logger.update('pi/entropy', -log_prob.mean().item())
        self.logger.update('pi/imitation', -IL_log_prob.mean().item())
        qval = q.predict(transitions.s, action)
        self.logger.update('pi/qval', qval.mean().item())
        self.logger.update('pi/advantage', (qval - b).mean().item())
        return loss

class EasyBCQLearner(PiLearner):
    def __init__(self, n_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.pinet = None

    def update_beta(self, beta):
        self.beta = beta

    def update_q(self, q):
        self.q = q
        
    def act(self, state, batch=False, sample=False):
        state = utils.torch_single_precision(state)
        state = state.to(self.device)
        if not batch:
            state = state.unsqueeze(0)

        # n_samples x batch_size x state_shape
        state = state.unsqueeze(0).repeat(self.n_samples, 1, 1)
        # (n_samples  batch_size) x state_shape
        state = state.reshape(-1, state.shape[-1])
        action = self.beta.pinet(state).sample().clamp(-1., 1.)

        qvals = self.q.predict(state, action)
        qvals = qvals.reshape(self.n_samples, -1)
        idx = qvals.argmax(dim=0)

        # n_sample x batch_size x action_shape
        action = action.reshape(self.n_samples, -1, action.shape[-1]).squeeze()
        # batch_size x action_shape
        final_action = action[idx, range(action.shape[1])]

        return final_action if batch else final_action.cpu().detach().numpy()

class ILRegLearner(PiLearner):
    def __init__(self, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
    
    def loss(self, transitions, q, baseline=None, beta=None):
        dist = self.pinet(transitions.s)
        action = dist.rsample()
        qval = q.predict(transitions.s, action)
        IL_log_prob = log_prob_func(dist, stable_action(transitions.a)) 
        loss = (- qval - self.alpha * IL_log_prob).mean()

        self.logger.update('pi/loss', loss.item())
        log_prob = log_prob_func(dist, action) 
        self.logger.update('pi/entropy', -log_prob.mean().item())
        self.logger.update('pi/imitation', -IL_log_prob.mean().item())
        self.logger.update('pi/qval', qval.mean().item())
        return loss


