import hydra
import os
import torch
import replay_buffer
import numpy as np
from copy import deepcopy
from experiment_logging import default_logger as logger


@hydra.main(config_path='config', config_name='train')
def train(cfg):
    print('jobname: ', cfg.name)

    # load data
    replay = torch.load(cfg.data_path)

    # load env
    import gym
    import d4rl
    if cfg.env.name[-2:] != 'v2' and cfg.env.name[-2:] != 'v1' \
        and cfg.env.name[-2:] != 'v0': 
        name_list = cfg.env.name.split('v2')
        env = gym.make(name_list[0] + 'v2')
    else:
        env = gym.make(cfg.env.name)
    cfg.state_dim = int(np.prod(env.observation_space.shape))
    cfg.action_dim = int(np.prod(env.action_space.shape))

    # set seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # build learners
    q = hydra.utils.instantiate(cfg.q)
    pi = hydra.utils.instantiate(cfg.pi)
    beta = hydra.utils.instantiate(cfg.beta)
    baseline = hydra.utils.instantiate(cfg.baseline)

    # setup logger 
    os.makedirs(cfg.log_dir, exist_ok=True)
    setup_logger(cfg)
    q.set_logger(logger)
    pi.set_logger(logger)
    beta.set_logger(logger)

    # train
    if cfg.pi.name == 'pi_easy_bcq':
        pi.update_beta(beta)
        pi.update_q(q)

    # train beta
    if cfg.train_beta:
        for step in range(int(cfg.betasteps)):
            beta.train_step(replay, None, None, None)

            if step % int(cfg.log_freq) == 0:
                logger.update('betatrain/step', step)
                beta.eval(env, cfg.eval_episodes)
                logger.write_sub_meter('betatrain')
            if step % int(cfg.beta_save_freq) == 0:
                beta.save(cfg.beta.model_save_path + '_' + str(step) + '.pt')

    # load beta as init pi
    pi.load_from_pilearner(beta)

    # iterate between eval and improvement
    for out_step in range(int(cfg.steps)):        
        # train Q
        if cfg.train_q:
            for in_step in range(int(cfg.qsteps)):
                step = out_step * int(cfg.qsteps) + in_step 
                
                q.train_step(replay, pi, beta)
                
                if step % q.target_update_freq == 0:
                    q.update_target()
                
                if step % int(cfg.log_freq) == 0:
                    logger.update('qtrain/step', step)
                    q.eval_env(env, pi, cfg.eval_episodes)
                    logger.write_sub_meter('qtrain')
                
                if step % int(cfg.q_save_freq) == 0:
                    q.save(cfg.q.model_save_path + '_' + str(step) + '.pt')

        # train pi
        if cfg.train_pi and cfg.pi.name != 'pi_easy_bcq':
            for in_step in range(int(cfg.pisteps)):
                pi.train_step(replay, q, baseline, beta)

                step = out_step * int(cfg.pisteps) + in_step
                if step % int(cfg.log_freq) == 0:
                    logger.update('pitrain/step', step)
                    pi.eval(env, cfg.eval_episodes)
                    logger.write_sub_meter('pitrain')
                if step % int(cfg.pi_save_freq) == 0:
                    pi.save(cfg.pi.model_save_path + '_' + str(step) + '.pt')
        elif cfg.pi.name == 'pi_easy_bcq':
            step = out_step + 1
            pi.update_q(q)
            if step % int(cfg.log_freq) == 0:
                logger.update('pitrain/step', step)
                pi.eval(env, cfg.eval_episodes)
                logger.write_sub_meter('pitrain')
    
    if cfg.train_q:
        q.save(cfg.q.model_save_path + '.pt')
    if cfg.train_pi:
        pi.save(cfg.pi.model_save_path + '.pt')

     
def setup_logger(cfg):
    logger_dict = dict()
    if cfg.train_q:
        q_train_dict = {'qtrain': {
                        'csv_path': f'{cfg.log_dir}/qtrain.csv',
                        'format_str': cfg.q.format_str,
                    },} 
        logger_dict.update(q_train_dict)
    if cfg.train_pi or cfg.pi.name == 'pi_easy_bcq':
        pi_train_dict = {'pitrain': {
                        'csv_path': f'{cfg.log_dir}/pitrain.csv',
                        'format_str': cfg.pi.format_str,
                    },} 
        logger_dict.update(pi_train_dict)
    if cfg.train_beta:
        beta_train_dict = {'betatrain': {
                        'csv_path': f'{cfg.log_dir}/betatrain.csv',
                        'format_str': cfg.beta.format_str,
                    },} 
        logger_dict.update(beta_train_dict)

    logger.setup(logger_dict, summary_format_str=None) 


if __name__ == "__main__":
    train()