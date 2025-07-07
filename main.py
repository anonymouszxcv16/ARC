import argparse
import os
import time
import gym
import numpy as np
import torch
import pybullet_envs

import TD

def train_offline(RL_agent, env, eval_env, args):
    # Performance.
    evals = []
    times = []

    # Values.
    ee_values = []
    cql_values, oe_values, ent_values = [], [], []
    negative_rewards = []

    # Load offline dataset.
    RL_agent.replay_buffer.load_D4RL(d4rl.qlearning_dataset(env))
    start_time = time.time()

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, times, ee_values, cql_values, oe_values, ent_values, negative_rewards, t, start_time, args)

        # Train.
        RL_agent.train()

# Train online RL agent.
def train_online(RL_agent, env, eval_env, args):
    # Reward
    evals = []

    # Time
    times = []

    # Values.
    ee_values = []
    cql_values, oe_values, ent_values = [], [], []
    negative_rewards = []

    # Initialize
    start_time = time.time()
    allow_train = False

    state, ep_finished = env.reset(), False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, times, ee_values, cql_values, oe_values, ent_values, negative_rewards, t, start_time, args)

        # Select action.
        if allow_train:
            action = RL_agent.select_action(np.array(state), deterministic=False if "SAC" in args.policy else True)
        else:
            action = env.action_space.sample()

        # Do a step.
        next_state, reward, ep_finished, _ = env.step(action)

        ep_total_reward += reward
        ep_timesteps += 1
        done = float(ep_finished) if ep_timesteps < env._max_episode_steps else 0

        # Store tuple.
        RL_agent.replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

        if allow_train:
            # Train.
            RL_agent.train()

        if ep_finished:

            if allow_train and args.use_checkpoints and "TD7" in args.policy:
                # TD7 UTD ratio training.
                RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

            if t >= args.timesteps_before_training:
                allow_train = True

            state, done = env.reset(), False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1


# Logs.
def maybe_evaluate_and_print(RL_agent, eval_env, evals, times, ee_values, cql_values, oe_values, ent_values, negative_rewards, t, start_time, args):
    if t % args.eval_freq == 0:
        # Rewards
        total_reward = np.zeros(args.eval_eps)
        discounted_reward = np.zeros(args.eval_eps)
        q_values = np.zeros(args.eval_eps)

        for ep in range(args.eval_eps):
            state, done = eval_env.reset(), False
            step = 0

            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float).to(args.device).unsqueeze(0)

                fixed_target_zs = RL_agent.fixed_encoder_target.zs(state)
                actor, _ = RL_agent.actor_target(state, fixed_target_zs, deterministic=False)
                fixed_target_zsa = RL_agent.fixed_encoder_target.zsa(fixed_target_zs, actor)

                q_values[ep] = RL_agent.critic_target(state, actor, fixed_target_zsa, fixed_target_zs).mean()

            # Episode
            while not done:
                # Action selection.
                action = RL_agent.select_action(state, args.use_checkpoints, use_exploration=False)

                # Step.
                state, reward, done, _ = eval_env.step(action)

                # Reward sum.
                discounted_reward[ep] += reward * RL_agent.args.discount ** step
                total_reward[ep] += reward

                step += 1

        # Time
        time_total = (time.time() - start_time) / 60

        # Reward
        score = eval_env.get_normalized_score(total_reward.mean()) * 100 if RL_agent.args.offline == 1 else total_reward.mean().item()

        # OE values.
        oe_value = (q_values - discounted_reward).mean().item()

        # EE values.
        ee_value = RL_agent.ee_value / args.eval_freq
        RL_agent.ee_value = 0

        # CQL values.
        cql_value = RL_agent.cql_value / args.eval_freq
        RL_agent.cql_value = 0

        # SAC values.
        ent_value = RL_agent.ent_value / args.eval_freq
        RL_agent.ent_value = 0

        negative_reward = (torch.sum(torch.tensor(RL_agent.replay_buffer.reward[:RL_agent.replay_buffer.size]) < 0, dim=0).item() /
                           max(RL_agent.replay_buffer.size, 1))
        negative_rewards.append(negative_reward)

        print(f"Timesteps: {(t + 1):,.1f}\tMinutes {time_total:.1f}\tRewards: {score:,.1f}\t"
              f"Entropy value: {ent_value:,.5f}\tCQL value: {cql_value:,.5f}\tEE value: {ee_value:,.5f}\tOE value: {oe_value:,.5f}\t"
              f"Negative Rewards: {negative_reward:,.5f}")

        # Reward
        evals.append(score)

        # Time
        times.append(time_total)

        # Values.
        ee_values.append(ee_value)
        cql_values.append(cql_value)
        oe_values.append(oe_value)
        ent_values.append(ent_value)

        # file.
        with open(f"./results/{args.env}/{args.file_name}", "w") as file:
            file.write(f"{evals}\n{times}\n{ee_values}\n{cql_values}\n{oe_values}\n{ent_values}\n{negative_rewards}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Algorithm.
    parser.add_argument("--policy", default="TD3", type=str)
    parser.add_argument("--alpha_sac", default=.01, type=float)
    parser.add_argument("--alpha_cql", default=.01, type=float)
    parser.add_argument("--alpha_ee", default=1, type=float)
    parser.add_argument("--alpha_srs", default=1, type=float)
    parser.add_argument("--alpha_rrs", default=1, type=float)
    parser.add_argument('--use_checkpoints', default=True)
    parser.add_argument('--offline', default=0, type=int)

    # Exploration.
    parser.add_argument("--timesteps_before_training", default=25_000, type=int)
    parser.add_argument("--exploration_noise", default=.1, type=float)
    parser.add_argument("--discount", default=.99, type=float)
    parser.add_argument("--N", default=2, type=int)
    parser.add_argument("--M", default=1, type=int)
    parser.add_argument("--buffer_size", default=1e6, type=int)

    # Environment.
    parser.add_argument("--env", default="HumanoidStandup-v2", type=str)
    # AntBulletEnv-v0, HalfCheetahBulletEnv-v0, HumanoidBulletEnv-v0, HopperBulletEnv-v0, Walker2DBulletEnv-v0, MinitaurBulletEnv-v0
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
        import d4rl

        d4rl.set_dataset_path(args.d4rl_path)
        args.use_checkpoints = False

    # environment
    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    # Seed.
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    eval_env.seed(args.seed + 100)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    RL_agent = TD.Agent(state_dim, action_dim, max_action, args)
    name = f"{args.policy}_{args.env}_{args.seed}"

    print("---------------------------------------")
    print(f"Algorithm: {args.policy}, Alpha SAC: {args.alpha_sac:,.3f}, Alpha CQL: {args.alpha_cql}, Alpha EE: {args.alpha_ee}, Alpha SRS: {args.alpha_srs}, "
          f"Alpha RRS: {args.alpha_rrs}, Buffer size: {args.buffer_size:,.1f}, Environment: {args.env}, Seed: {args.seed}, Device: {RL_agent.device}")
    print("---------------------------------------")

    # Optimize.
    if args.offline == 1:
        train_offline(RL_agent, env, eval_env, args)
    else:
        train_online(RL_agent, env, eval_env, args)