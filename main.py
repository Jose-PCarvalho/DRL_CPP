import argparse
import bz2
import random
from datetime import datetime
import os
import pickle

import numpy as np
import torch
import yaml
from tqdm import trange
from src.Environment.Environment import *
from src.Rainbow.agent import *
from src.Rainbow.memory import ReplayMemory
from test import test
from src.Environment.BackAndForth import *
from src.Environment.WallFollowing import *


def log(s, log_dir=None):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)
    if dir is not None:
        with open(log_dir, 'a') as file:
            file.write(('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s + '\n'))


def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


def use_pseudo(ovrl):
    p = -1.4 * (1 - ovrl) + 0.7
    eps = np.random.random()
    if eps < p:
        return True
    else:
        return False


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--id', type=str, default='CPP', help='Experiment ID')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--T-max', type=int, default=int(70e4), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--history-length', type=int, default=3, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='data-efficient', choices=['canonical', 'data-efficient'],
                    metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-625, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=625, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(150000), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=5, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(70e3), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=10000, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N',
                    help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=250, metavar='N',
                    help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=50000,
                    help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true',
                    help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
parser.add_argument('--config-file', type=str, default='configs/training_obstacles.yaml')
parser.add_argument('--log-file', type=str, default='results/log.txt')
parser.add_argument('--starting-episode',type=int, default=3)

# Setup
args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf'), 'overlap': []}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
    print("Initiating cuda")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = args.enable_cudnn
else:
    args.device = torch.device('cpu')
    print("Initiating CPU")

with open(args.config_file, 'rb') as f:
    conf = yaml.safe_load(f.read())  # load the config file

if args.log_file:
    with open(args.log_file, 'w') as file:
        pass

number_envs = len(conf.keys())
env = Environment(EnvironmentParams(conf['env1']))
action_space = env.action_space()
starting_priority_weight = args.priority_weight
dqn = Agent(args, action_space)

# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
# if args.model is not None and not args.evaluate:
#     if not args.memory:
#         raise ValueError('Cannot resume training without memory save path. Aborting...')
#     elif not os.path.exists(args.memory):
#         raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))
#
#     mem = load_memory(args.memory, args.disable_bzip_memory)
#
# else:

mem = ReplayMemory(args, args.memory_capacity)
avg_overlap = 0
retries = 0
all_T = 0
T = 0
e = args.starting_episode
while e < number_envs + 1:

    if e > 1 and avg_overlap > 0.2:
        e -= 1
        args.priority_weight = (1 - starting_priority_weight) / (retries + 1) + starting_priority_weight
        retries += 1
    else:
        print(avg_overlap)
        args.priority_weight = starting_priority_weight
        priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
        retries = 0
        args.T_max = conf['env' + str(e)]['base_steps']
        mem = ReplayMemory(args, args.memory_capacity)
    env_str = 'env' + str(e)
    env = Environment(EnvironmentParams(conf[env_str]))
    print(conf[env_str])
    non_obstacles = conf[env_str]['number_obstacles'] == 0
    dqn.update_C51(conf[env_str]['size'])
    use_pseudo_agent, pseudo_episode = non_obstacles, non_obstacles
    val_mem = ReplayMemory(args, args.evaluation_size)
    all_T += T
    T, done, truncated = 0, True, True
    avg_overlap = 1
    while T < args.evaluation_size:

        if done or truncated:
            state, info = env.reset()

        next_state, _, done, truncated, info = env.step(np.random.randint(0, action_space))
        val_mem.append(state[0], state[1], -1, 0.0, done or truncated)
        state = next_state
        T += 1

    if args.evaluate:
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q, avg_overlap = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True,
                                              env_args=conf[env_str])  # Test
        print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))

    else:
        # Training loop
        dqn.train()
        done = True
        truncated = True
        for T in trange(1, args.T_max + 1):

            if done or truncated:
                state, _ = env.reset()
                pseudo_episode = False
                if use_pseudo_agent and avg_overlap > 0.5:
                    pseudo_episode = use_pseudo(avg_overlap)
                    if pseudo_episode:
                        alg = random.choice([BackForth, WallFollower])
                        pseudo_agent = alg()
                        pseudo_agent.init(state[0][-1], env.state.params.size)

            if T % args.replay_frequency == 0:
                dqn.reset_noise()  # Draw a new set of noisy weights

            if not pseudo_episode:
                action = dqn.act(state[0], state[1])
            else:
                action = pseudo_agent.select_action(state[0][-1]).value

            next_state, reward, done, truncated, info = env.step(action)  # Step

            mem.append(state[0], state[1], action, reward, done or truncated)  # Append transition to memory

            # Train and test
            if T >= args.learn_start:

                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

                if T % args.replay_frequency == 0:
                    dqn.learn(mem)  # Train with n-step distributional double-Q learning

                if T % args.evaluation_interval == 0:
                    dqn.eval()  # Set DQN (online network) to evaluation mode
                    avg_reward, avg_Q, avg_overlap = test(args, T + all_T, dqn, val_mem, metrics, results_dir,
                                                          env_args=conf[env_str])  # Test
                    log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | env: ' + conf[env_str]['name'] +
                        ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q) + ' | Avg. Overlap: ' + str(
                        avg_overlap), args.log_file)
                    dqn.train()  # Set DQN (online network) back to training mode

                    # If memory path provided, save it
                    if args.memory is not None:
                        save_memory(mem, args.memory, args.disable_bzip_memory)

                # Update target network
                if T % args.target_update == 0:
                    dqn.update_target_net()

                # Checkpoint the network
                if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                    dqn.save(results_dir, 'checkpoint.pth')

            state = next_state
        e += 1
        dqn.save(results_dir, conf[env_str]['name'] + '.pth')

    # env.close()
