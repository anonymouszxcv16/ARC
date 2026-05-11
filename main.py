import argparse
import os
import time
import gym
import numpy as np
import torch
import random

from collections import defaultdict

import TD

def train_offline(RL_agent, env, eval_env, args):
    # Performance.
    evals = []
    times = []

    mus = []
    Ms = []

    alphas = []
    stds = []

    # Load offline dataset.
    start_time = time.time()

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, times, alphas, stds, mus, Ms, t, start_time, args)

        # Train.
        RL_agent.train()

# Train online RL agent.
def train_online(RL_agent, env, eval_env, args):
    # Reward
    evals = []
    times = []
    alphas = []
    stds = []

    mus, Ms = [], []

    # Initialize
    start_time = time.time()
    allow_train = False

    state, ep_finished = env.reset(), False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    # SASR
    RL_agent.replay_buffer.success_count = defaultdict(int)
    RL_agent.replay_buffer.failure_count = defaultdict(int)

    episode_transitions = []

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, times, alphas, stds, mus, Ms, t, start_time, args)

        # Select action.
        if allow_train:
            action = RL_agent.select_action(np.array(state), deterministic=False if "SAC" in args.policy else True)
        else:
            action = env.action_space.sample()

        # Do a step.
        next_state, reward, ep_finished, _ = env.step(action)

        if "Noise" in args.policy:
            sign = 1 if random.randint(0, 1) == 1 else -1
            noise = reward * random.random() * sign * RL_agent.args.noise_frac_max
            reward += noise

        # SASR
        episode_transitions.append(state)

        ep_total_reward += reward
        ep_timesteps += 1
        done = float(ep_finished or ep_timesteps == env._max_episode_steps)

        # Store tuple.
        RL_agent.replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

        if allow_train:
            # Train.
            RL_agent.train()

        if done:
            if t >= args.timesteps_before_training:
                allow_train = True

            # SASR
            label = 1 if reward > 0 else 0
            # label = 1 if ep_total_reward > 0 else 0

            for s in episode_transitions:
                if label == 1:
                    RL_agent.replay_buffer.success_count[RL_agent.state_key(s)] += 1
                else:
                    RL_agent.replay_buffer.failure_count[RL_agent.state_key(s)] += 1

            state, done = env.reset(), False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1

            episode_transitions = []


# Logs.
def maybe_evaluate_and_print(RL_agent, eval_env, evals, times, alphas, stds, mus, Ms, t, start_time, args):
    if t % args.eval_freq == 0:
        # Rewards
        total_reward = np.zeros(args.eval_eps)

        for ep in range(args.eval_eps):
            # gymnasium
            state, done = eval_env.reset()[0], False
            step = 0

            # Episode
            while not done:
                # Action selection.
                action = RL_agent.select_action(state, use_exploration=False)

                # Step.
                state, reward, done, trunc, _ = eval_env.step(action)

                # Reward sum.
                total_reward[ep] += reward

                step += 1

                # gymnasium
                done = done or trunc

        # Time
        time_total = (time.time() - start_time) / 60
        score = minari.get_normalized_score(RL_agent.dataset, total_reward.mean()) if RL_agent.args.offline == 1 else total_reward.mean().item()
        std = RL_agent.rewards_std / args.eval_freq

        mu = RL_agent.replay_buffer.reward.mean().item()
        M = RL_agent.replay_buffer.reward.max().item()

        print(f"Timesteps: {(t + 1):,.1f}\tMinutes {time_total:.1f}\tRewards: {score:,.1f}\tAlpha: {RL_agent.args.alpha:,.3f}\tRewards Std: {std:,.3f}\t"
              f"mu: {mu:,.2f}\tM: {M:,.2f}")

        evals.append(score)
        times.append(time_total)
        alphas.append(RL_agent.args.alpha)
        stds.append(std)

        mus.append(mu)
        Ms.append(M)

        # file.
        with open(f"./results/{args.env}/{args.file_name}", "w") as file:
            file.write(f"{evals}\n{times}\n{alphas}\n{stds}\n{mus}\n{M}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Algorithm.
    parser.add_argument("--policy", default="TD3", type=str)
    parser.add_argument("--alpha_cql", default=5, type=float)
    parser.add_argument("--noise_frac_max", default=.01, type=float)
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--auto_alpha", default=0, type=int)
    parser.add_argument("--auto_alpha_interval", default=100_000, type=int)
    parser.add_argument('--offline', default=0, type=int)

    # Exploration.
    parser.add_argument("--timesteps_before_training", default=25_000, type=int)
    parser.add_argument("--exploration_noise", default=.1, type=float)
    parser.add_argument("--discount", default=.99, type=float)
    parser.add_argument("--N", default=2, type=int)
    parser.add_argument("--buffer_size", default=1e6, type=int)

    # Environment.
    parser.add_argument("--env", default="HumanoidStandup-v2", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)

    # Evaluation
    parser.add_argument("--eval_freq", default=5_000, type=int)
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)

    # File
    parser.add_argument('--file_name', default=None)
    args = parser.parse_args()

    if args.file_name is None:
        args.file_name = f"{args.policy}_{args.seed}"

    if not os.path.exists(f"./results/{args.env}"):
        os.makedirs(f"./results/{args.env}")

    # Offline.
    if args.offline == 1:
        import minari

        dataset = minari.load_dataset(f"{args.env}", download=True)
        env = dataset.recover_environment()
        eval_env = dataset.recover_environment()

        args.use_checkpoints = False

    else:
        # environment
        env = gym.make(args.env)
        eval_env = gym.make(args.env)

    # Seed.
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    RL_agent = TD.Agent(state_dim, action_dim, max_action, args)
    name = f"{args.policy}_{args.env}_{args.seed}"

    if args.offline == 1:
        RL_agent.dataset = dataset
        RL_agent.replay_buffer.load_D4RL(dataset)

    print("---------------------------------------")
    print(f"Algorithm: {args.policy}, Alpha: {args.alpha}, Buffer size: {args.buffer_size:,.1f}, "
          f"Environment: {args.env}, Seed: {args.seed}, Device: {RL_agent.device}")
    print("---------------------------------------")

    # Optimize.
    if args.offline == 1:
        train_offline(RL_agent, env, eval_env, args)
    else:
        train_online(RL_agent, env, eval_env, args)