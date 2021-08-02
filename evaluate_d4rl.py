import torch
import numpy as np
import hydra

# import os
# os.environ['D4RL_DATASET_DIR'] = 

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

def get_path(env_name, pi_name, hyper_type, hyper_val, alg_name, seed):

    q_path = '/models/q_' + env_name + '_width_1024_seed_' + str(seed) + '_q.pt'

    if (pi_name == 'pi_easy_bcq' or pi_name == 'beta') and alg_name == 'onestep':
        path = '/models/beta_' + env_name + '_dist_squash_width_1024_seed_' \
                    + str(seed) + '_beta.pt'
        
    elif alg_name == 'onestep':
        path = '/models/onestep_init_' + pi_name + '_' + hyper_type + '_' + \
                    str(hyper_val) + '_' + env_name + '_seed_' + str(seed) + '_pi.pt' 
    
    elif alg_name == 'multistep':
        path = '/models/multistep_'+ env_name + '_' + pi_name + \
                 '_' + hyper_type + '_' + str(hyper_val) + \
                     '_seed_' + str(seed) + '_pi.pt' 

    elif alg_name == 'iterative':
        path = '/models/iterative_init_'+ env_name + '_' + pi_name + \
                 '_' + hyper_type + '_' + str(hyper_val) + '_seed_' + \
                      str(seed) + '_pi.pt' 
    
    return path, q_path



hyper_dict = {
    'pi_reverse_kl': {'alpha': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}, 
    'pi_marwil': {'temp': [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]}, 
    'pi_easy_bcq': {'n': [2, 5, 10, 20, 50, 100]},
    'beta': {'n': [1]}
}

env_list = ['halfcheetah-medium-v2', 'halfcheetah-medium-expert-v2',
            'halfcheetah-random-v2', 'walker2d-medium-v2', 
            'walker2d-medium-expert-v2', 'walker2d-random-v2', 
            'hopper-medium-v2', 'hopper-medium-expert-v2',
            'hopper-random-v2', 'pen-cloned-v1', 'hammer-cloned-v1',
            'door-cloned-v1', 'relocate-cloned-v1',
            'halfcheetah-medium-replay-v2', 'walker2d-medium-replay-v2', 'hopper-medium-replay-v2']


@hydra.main(config_path='config', config_name='eval')
def eval(cfg):
    
    env_name = cfg.env_name
    pi_name = cfg.pi_name
    alg_name = cfg.alg_name
    root_path = cfg.path
    seed = cfg.seed
    
    qsteps = '2e6'
    betasteps = '1e6' 
    n_episodes = 100

    print(env_name)
    import gym
    import d4rl
    if env_name[-2:] != 'v2' and env_name[-2:] != 'v1' \
            and env_name[-2:] != 'v0': 
        name_list = env_name.split('v2')
        env = gym.make(name_list[0] + 'v2')
    else:
        env = gym.make(env_name)
    env.seed(seed)
    
    returns_dict = {}
    #returns_dict = torch.load('results/' + env_name)

    print (pi_name)
    
    hyper_val_dict = hyper_dict[pi_name]
    hyper_type = list(hyper_val_dict.keys())[0]
    hyper_val_list = list(hyper_val_dict.values())[0]
    
    for hyper_val in hyper_val_list:
        print(hyper_val)
        path, q_path = get_path(env_name, pi_name, hyper_type, 
                                hyper_val, alg_name, seed)
        
        if pi_name == 'pi_easy_bcq':
            pi = get_bcq(hyper_val, root_path + path, 
                            root_path + q_path, 'squash', env)
        else:
            pi = get_policy(root_path + path, 'squash', env)
        
        norm_returns, returns = run_policy(env, pi, n_episodes)
        returns_dict[str(hyper_val)] = norm_returns

    torch.save(returns_dict, root_path + '/results/' + alg_name +
                            '_' + pi_name + '_' + env_name + '_' + str(seed))


if __name__ == "__main__":
    eval()

        