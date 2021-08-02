import torch
from torch import nn
from torch.nn import functional as F
import utils
import torch.distributions as D

def soft_clamp(x, low, high):
    x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x


class GaussMLP(nn.Module):
    def __init__(self, state_dim, action_dim, width, depth, dist_type):
        super().__init__()
        self.net = utils.MLP(input_shape=(state_dim), output_dim=2*action_dim,
                        width=width, depth=depth)
        self.log_std_bounds = (-5., 0.)
        self.mu_bounds = (-1., 1.)
        self.dist_type = dist_type
    
    def forward(self, s):
        s = torch.flatten(s, start_dim=1)
        mu, log_std = self.net(s).chunk(2, dim=-1)
        
        mu = soft_clamp(mu, *self.mu_bounds)
        log_std = soft_clamp(log_std, *self.log_std_bounds)

        std = log_std.exp()
        if self.dist_type == 'normal':
            dist = D.Normal(mu, std)
        elif self.dist_type == 'trunc':
            dist = utils.TruncatedNormal(mu, std)
        elif self.dist_type == 'squash':
            dist = utils.SquashedNormal(mu, std)
        else:
            raise TypeError("Expected dist_type to be 'normal', 'trunc', or 'squash'")
        return dist


class MixedGaussMLP(nn.Module):
    def __init__(self, n_comp, state_dim, action_dim, width, depth):
        super().__init__()
        self.n_comp = n_comp
        self.action_dim = action_dim
        self.net = utils.MLP(input_shape=(state_dim), 
                        output_dim=(2*n_comp)*action_dim + n_comp,
                        width=width, depth=depth)
        self.log_std_bounds = (-5., 0.)
        self.mu_bounds = (-2., 2.)
    
    def forward(self, s):
        s = torch.flatten(s, start_dim=1)
        preds = self.net(s)
        cat = F.softmax(preds[:, :self.n_comp], dim=-1)
        mus = preds[:, self.n_comp: self.n_comp + self.n_comp*self.action_dim]
        log_stds = preds[:, self.n_comp + self.n_comp*self.action_dim:]
        
        # mus is  batch x n_comp x a_dim
        mus = soft_clamp(mus, *self.mu_bounds)
        log_stds = soft_clamp(log_stds, *self.log_std_bounds)
        stds = log_stds.exp().transpose(0,1)

        mus = mus.reshape(-1, self.n_comp, self.action_dim)
        stds = stds.reshape(-1, self.n_comp, self.action_dim)
        
        mix = D.Categorical(cat)
        #comp = D.Independent(utils.SquashedNormal(mus, stds), 1)
        #comp = D.Independent(utils.SquashedNormal(mus, stds), 1)
        comp = D.Independent(D.Normal(mus, stds), 1)
        dist = D.MixtureSameFamily(mix, comp)
        
        return dist

