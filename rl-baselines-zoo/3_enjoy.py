import os
import sys
import argparse
import importlib
import warnings
import gym_envs  # added by Pierre
import pandas as pd # added by Pierre
import time # added by Pierre

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import utils.import_envs  # pytype: disable=import-error
import numpy as np
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv

from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model
from utils.utils import StoreDict

# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
                        type=int)
    parser.add_argument('--n-envs', help='number of environments', default=1,
                        type=int)
    parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=-1,
                        type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--no-render', action='store_true', default=False,
                        help='Do not render the environment (useful for tests)')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', default=False,
                        help='Use stochastic actions (for DDPG/DQN/SAC)')
    parser.add_argument('--load-best', action='store_true', default=False,
                        help='Load best model instead of last model if available')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='Normalize reward if applicable (trained with VecNormalize)')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict, help='Optional keyword argument to pass to the env constructor')
    parser.add_argument('--render-pybullet', help='Slow down Pybullet simulation to render', default=False) # added by Pierre
    parser.add_argument('--random-pol', help='Random policy', default=False) # added by Pierre
    parser.add_argument('--log-dir-random', help='Log directory of the random policy') # added by Pierre
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print('Loading latest experiment, id={}'.format(args.exp_id))

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, args.exp_id))
    else:
        log_path = os.path.join(folder, algo)


    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    if not args.random_pol:  # added by Pierre
        model_path = find_saved_model(algo, log_path, env_id, load_best=args.load_best)

    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        args.n_envs = 1

    set_global_seeds(args.seed)

    is_atari = 'NoFrameskip' in env_id

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    log_dir = args.reward_log if args.reward_log != '' else None

    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs

    env = create_test_env(env_id, n_envs=args.n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                          should_render=not args.no_render,
                          hyperparams=hyperparams, env_kwargs=env_kwargs)

    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env

    if not args.random_pol:  # added by Pierre
        model = ALGOS[algo].load(model_path, env=load_env)

    obs = env.reset()

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not args.stochastic

    # INITIALISE METRICS
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    success_threshold_50 = 0.05
    success_list_50, reachtime_list_50, episode_success_list_50 = [], [], []
    success_threshold_20 = 0.02
    success_list_20, reachtime_list_20, episode_success_list_20 = [], [], []
    success_threshold_10 = 0.01
    success_list_10, reachtime_list_10, episode_success_list_10 = [], [], []
    success_threshold_5 = 0.005
    success_list_5, reachtime_list_5, episode_success_list_5 = [], [], []

    # For HER, monitor success rate
    successes = []
    state = None

    for _ in range(args.n_timesteps):

        # Added by Pierre
        if args.random_pol:
            action = [env.action_space.sample()] # Random Agent
        else:
            action, state = model.predict(obs, state=state, deterministic=deterministic)

        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)

        # if args.render_pybullet:
        #     time.sleep(1./30.)     # added by Pierre (slow down Pybullet for rendering)

        # added by Pierre
        if infos[0]['dist_ft_t'] <= success_threshold_50:
            episode_success_list_50.append(1)
        else:
            episode_success_list_50.append(0)

        if infos[0]['dist_ft_t'] <= success_threshold_20:
            episode_success_list_20.append(1)
        else:
            episode_success_list_20.append(0)

        if infos[0]['dist_ft_t'] <= success_threshold_10:
            episode_success_list_10.append(1)
        else:
            episode_success_list_10.append(0)

        if infos[0]['dist_ft_t'] <= success_threshold_5:
            episode_success_list_5.append(1)
        else:
            episode_success_list_5.append(0)

        if not args.no_render:
            env.render('human')
            # env.render(mode="human") 

        episode_reward += reward[0]
        ep_len += 1

        if args.n_envs == 1:
            # For atari the return reward is not the atari score
            # so we have to get it from the infos dict
            if is_atari and infos is not None and args.verbose >= 1:
                episode_infos = infos[0].get('episode')
                if episode_infos is not None:
                    print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
                    print("Atari Episode Length", episode_infos['l'])

            if done and not is_atari and args.verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print("Episode Reward: {:.2f}".format(episode_reward))
                print("Episode Length", ep_len)
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)

                # Pierre: append the last element of the episode success list when episode is done
                success_list_50.append(episode_success_list_50[-1]) 
                success_list_20.append(episode_success_list_20[-1]) 
                success_list_10.append(episode_success_list_10[-1]) 
                success_list_5.append(episode_success_list_5[-1])  

                # if the episode is successful and it starts from an unsucessful step, calculate reach time
                if episode_success_list_50[-1] == True and episode_success_list_50[0] == False:
                    idx = 0
                    while episode_success_list_50[idx] == False:
                        idx += 1
                    reachtime_list_50.append(idx)

                if episode_success_list_20[-1] == True and episode_success_list_20[0] == False:
                    idx = 0
                    while episode_success_list_20[idx] == False:
                        idx += 1
                    reachtime_list_20.append(idx)

                if episode_success_list_10[-1] == True and episode_success_list_10[0] == False:
                    idx = 0
                    while episode_success_list_10[idx] == False:
                        idx += 1
                    reachtime_list_10.append(idx)

                if episode_success_list_5[-1] == True and episode_success_list_5[0] == False:
                    idx = 0
                    while episode_success_list_5[idx] == False:
                        idx += 1
                    reachtime_list_5.append(idx)

                # RESET FOR NEW EPISODE
                state = None
                episode_reward = 0.0
                ep_len = 0
                episode_success_list_50 = []  
                episode_success_list_20 = []  
                episode_success_list_10 = []  
                episode_success_list_5 = []  

            # Reset also when the goal is achieved when using HER
            if done or infos[0].get('is_success', False):
                if args.algo == 'her' and args.verbose > 1:
                    print("Success?", infos[0].get('is_success', False))
                # Alternatively, you can add a check to wait for the end of the episode
                # if done:
                obs = env.reset()
                if args.algo == 'her':
                    successes.append(infos[0].get('is_success', False))
                    episode_reward, ep_len = 0.0, 0

    if args.verbose > 0 and len(successes) > 0:
        print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f} +/- {:.2f}".format(np.mean(episode_rewards), np.std(episode_rewards)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(success_threshold_50, np.mean(success_list_50), np.mean(reachtime_list_50)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(success_threshold_20, np.mean(success_list_20), np.mean(reachtime_list_20)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(success_threshold_10, np.mean(success_list_10), np.mean(reachtime_list_10)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(success_threshold_5, np.mean(success_list_5), np.mean(reachtime_list_5)))

    if args.verbose > 0 and len(episode_lengths) > 0:
        print("Mean episode length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))


    # added by Pierre
    print("path:", log_path)
    d = {
        "Eval mean reward": np.mean(episode_rewards), 
        "Eval std": np.std(episode_rewards), 
        "success ratio 50mm": np.mean(success_list_50),
        "Average reach time 50mm": np.mean(reachtime_list_50),
        "success ratio 20mm": np.mean(success_list_20),
        "Average reach time 20mm": np.mean(reachtime_list_20),
        "success ratio 10mm": np.mean(success_list_10),
        "Average reach time 10mm": np.mean(reachtime_list_10),
        "success ratio 5mm": np.mean(success_list_5),
        "Average reach time 5mm": np.mean(reachtime_list_5),
        }
    df = pd.DataFrame(d, index=[0])

    if args.random_pol:
        log_rand = args.log_dir_random
        df.to_csv(log_rand+"/stats.csv", index=False)
    else:
        df.to_csv(log_path+"/stats.csv", index=False)


    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            # SubprocVecEnv
            env.close()


if __name__ == '__main__':
    main()
