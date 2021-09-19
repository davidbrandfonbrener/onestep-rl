import torch
import numpy as np
import hydra
import os

import policy_learner, q_learner, baseline_learner

def get_policy(path, dist, env, ncomp=1): 
    pi = policy_learner.PiLearner(env.observation_space.shape[0],
                                env.action_space.shape[0],
                                dist=dist, n_comp=ncomp,
                                width=1024)
    pi.pinet.load_state_dict(torch.load(path, map_location=pi.device))
    return pi

def get_q(path, env): 
    q = q_learner.QLearner(env.observation_space.shape[0],
                                env.action_space.shape[0],
                                width=1024)
    q.qnet.load_state_dict(torch.load(path, map_location=q.device))
    return q

def get_bcq(n_samples, beta_path, q_path, dist, env):
    pi = policy_learner.EasyBCQLearner(n_samples=n_samples,
                                    state_dim=env.observation_space.shape[0],
                                    action_dim=env.action_space.shape[0])
    q = get_q(q_path, env)
    beta = get_policy(beta_path, dist, env)
    pi.update_beta(beta)
    pi.update_q(q)
    return pi

def run_policy(env, pi, n_episodes):
    returns = np.zeros(n_episodes)
    for episode in range(n_episodes):
        done = False
        state = env.reset()
        step, ep_return = 0, 0
        while not done:
            action = pi.act(state, batch=False, sample=False)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_return += reward
            step += 1
        returns[episode] = ep_return
    norm_returns = env.get_normalized_score(returns)
    return norm_returns, returns


@hydra.main(config_path='config', config_name='eval')
def eval(cfg):
    root_path = cfg.path
    pi_path = os.path.join(root_path, cfg.pi_path)
    q_path = os.path.join(root_path, cfg.q_path)

    import gym
    import d4rl
    if cfg.env_name[-2:] != 'v2' and cfg.env_name[-2:] != 'v1' \
            and cfg.env_name[-2:] != 'v0': 
        name_list = cfg.env_name.split('v2')
        env = gym.make(name_list[0] + 'v2')
    else:
        env = gym.make(cfg.env_name)
    env.seed(cfg.seed)
    
    if cfg.pi_name == 'easy_bcq':
        pi = get_bcq(cfg.n_bcq_samples, pi_path, q_path, cfg.dist, env)
    else:
        pi = get_policy(pi_path, cfg.dist, env)
    
    norm_returns, returns = run_policy(env, pi, cfg.n_episodes)

    save_path = root_path + '/results/' + cfg.env_name +\
                            '_' + cfg.pi_name + '_' + cfg.pi_path + '_' + str(cfg.seed)
    torch.save(norm_returns, save_path)
    print('Results saved at: ' + save_path)


if __name__ == "__main__":
    eval()