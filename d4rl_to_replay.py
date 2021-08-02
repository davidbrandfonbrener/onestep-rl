import d4rl
import gym
import torch
import numpy as np
import argparse
import replay_buffer

# import os
# os.environ['D4RL_DATASET_DIR'] = 

def d4rl_to_replay(name):
    env = gym.make(name)
    dataset = env.get_dataset()

    replay = replay_buffer.Replay(env.observation_space.shape, env.action_space.shape,
                                max_size = len(dataset['rewards']),
                                has_next_action=True)

    for i in range(len(dataset['rewards']) - 1):
        if not dataset['timeouts'][i]:
            s = dataset['observations'][i]
            a = dataset['actions'][i]
            r = dataset['rewards'][i]
            term = dataset['terminals'][i]
            sp = dataset['observations'][i+1]
            ap = dataset['actions'][i+1]
            transition = replay_buffer.Transition(s,a,r,sp,ap, done=term)
            replay.append(transition)
    
    print("Replay length: ", len(replay))
    path = 'data/' + name + '.pt'
    torch.save(replay, path)
    print("Successfully saved at: ", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='halfcheetah-random-v2')
    args = parser.parse_args()

    d4rl_to_replay(args.name)