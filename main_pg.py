import argparse
import os
import time
import numpy as np
import torch
import TD

# Train online RL agent.
def train_online(RL_agent, env, eval_env, args):
    # Reward
    evals = []
    biases = []
    variances = []

    # Time
    times = []

    # Loss
    losses = []

    # Policy
    policy_best = []

    # Initialize
    start_time = time.time()
    allow_train = False

    state, ep_finished = env.reset()[0], False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    state = state['observation']

    # Gt
    t0 = 0

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        RL_agent.t += 1
        maybe_evaluate_and_print(RL_agent, eval_env, evals, biases, variances, times, losses, policy_best, t, start_time, args)

        # Select action.
        if allow_train:
            action = RL_agent.select_action(np.array(state), deterministic=False)
        else:
            action = env.env.action_space.sample()

        # Do a step.
        next_state, reward, done, trunc, _ = env.step(action)

        ep_total_reward += reward
        ep_timesteps += 1
        ep_finished = float(done or trunc)

        next_state = next_state['observation']

        # Store tuple.
        RL_agent.experience_replay.add(torch.tensor(state).unsqueeze(0), torch.tensor(action), torch.tensor(next_state),
                                      reward, done)

        state = next_state

        if allow_train and (not "REINFORCE" in args.policy or ep_finished):
            for _ in range(args.UTD):
                # Train.
                RL_agent.train()

        if ep_finished:
            if t >= args.timesteps_before_training:
                allow_train = True

            state, done = env.reset()[0], False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1

            state = state['observation']

            # Episode length
            T = RL_agent.t - t0
            scores = torch.zeros((T, 1)).to(args.device)

            # Gt
            for step in range(T):
                # t
                for t in range(step, T):
                    # r_t
                    r_t = RL_agent.experience_replay.reward[RL_agent.t - t0 + t]

                    if "RRS" in args.policy or "NRS" in args.policy and RL_agent.experience_replay.size > 0:
                        eps = 1e-32

                        # normalization
                        r_mean, r_max = RL_agent.experience_replay.reward[
                                                    :RL_agent.experience_replay.size].mean(), RL_agent.experience_replay.reward[
                                                                                      :RL_agent.experience_replay.size].max()

                        r_t = ((r_t - r_mean + eps) / (r_max + eps))

                        if "RRS" in args.policy:
                            # activation
                            r_t = (1 / args.alpha) * 1 / (1 + (-args.alpha * r_t).exp()) * (1 + (args.alpha * r_t).exp()).log()

                            if args.auto_alpha == 1:
                                rs_std = RL_agent.experience_replay.reward.std()
                                std_inverse = 1 / rs_std
                                RL_agent.std_inverse_sum += std_inverse

                                if (RL_agent.training_steps + args.timesteps_before_training) % args.auto_alpha_interval == 0:
                                    std_inverse_mean = RL_agent.std_inverse_sum / args.auto_alpha_interval
                                    args.alpha = RL_agent.scaled_sigmoid(std_inverse_mean.cpu())
                                    RL_agent.std_inverse_sum = 0

                    # r_t ** discount
                    scores[step] += r_t * args.discount ** (t - step)

            RL_agent.experience_replay.mc_score[t0:RL_agent.t] = scores
            t0 = RL_agent.t

# Logs.
def maybe_evaluate_and_print(RL_agent, eval_env, evals, biases, variances, times, losses, policy_best, t, start_time, args):
    if t % args.eval_freq == 0:
        # Rewards
        q_values = np.zeros(args.eval_eps)
        discounted_reward = np.zeros(args.eval_eps)

        for ep in range(args.eval_eps):
            state = eval_env.reset()
            state = state[0]['observation']

            done = False
            action = RL_agent.select_action(state, deterministic=True)

            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float).to(RL_agent.args.device).unsqueeze(0)
                action = torch.tensor(action, dtype=torch.float).to(RL_agent.args.device).unsqueeze(0)
                Q_target = RL_agent.critic_target(state, action, ).min(1, keepdim=True)[0]

            q_values[ep] = Q_target
            step = 0

            if ep == 0:
                policy = []

            # Episode
            while not done:
                # Action selection.
                action = RL_agent.select_action(state, deterministic=True)

                if ep == 0:
                    policy.append(action)

                # Step.
                state, reward, done, trunc, _ = eval_env.step(action)
                done = done or trunc
                state = state['observation']

                # Reward sum.
                discounted_reward[ep] += reward * RL_agent.args.discount ** step

                step += 1

        # Time
        time_total = (time.time() - start_time) / 60

        # Reward
        score = discounted_reward.mean().item()
        q_score = q_values.mean().item()
        bias = torch.tensor(score - q_score).abs().item()
        variance = discounted_reward.std().item()

        # Loss
        loss_tot = RL_agent.estimate_loss(replay=RL_agent.experience_replay) if not "REINFORCE" in args.policy else RL_agent.loss_tot / args.eval_freq
        RL_agent.loss_tot = 0

        print(f"Timesteps: {(t + 1):,.1f}\tMinutes {time_total:.1f}\tScore: {score:,.1f}\tQ-Score: {q_score:,.1f}\tBias: {bias:,.1f}\t"
              f"Variance: {variance:,.1f}\t"
              f"Loss: {loss_tot:,.1f}\t")

        # Reward
        evals.append(score)
        biases.append(bias)
        variances.append(variance)

        # Time
        times.append(time_total)

        # Loss
        losses.append(loss_tot)

        # Policy
        if score == max(evals):
            policy_best = policy

        # file.
        with open(f"./results/{args.env}/{args.file_name}", "w") as file:
            file.write(f"{evals}\n{times}\n{losses}\n{biases}\n{variances}\n"
                       f"{policy_best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Algorithm.
    parser.add_argument("--policy", default="DDPG", type=str)
    parser.add_argument('--offline', default=0, type=int)

    # Exploration.
    parser.add_argument("--timesteps_before_training", default=25_000, type=int)
    parser.add_argument("--exploration_noise", default=.1, type=float)
    parser.add_argument("--discount", default=.99, type=float)
    parser.add_argument("--N", default=1, type=int)
    parser.add_argument("--UTD", default=1, type=int)

    # ppo
    parser.add_argument("--eps", default=0.2, type=float)

    parser.add_argument("--buffer_size", default=1e6, type=int)

    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--auto_alpha", default=1, type=int)
    parser.add_argument("--auto_alpha_interval", default=100_000, type=int)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)

    # Environment.
    parser.add_argument("--env", default="PointMaze_UMaze-v3", type=str)

    # Evaluation
    parser.add_argument("--eval_freq", default=5_000, type=int)
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)

    # File
    parser.add_argument('--file_name', default=None)
    args = parser.parse_args()

    if args.file_name is None:
        args.file_name = f"{args.policy}_{args.seed}"

    import sys

    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # strip all args

    import gymnasium as gym  # safe import
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    # environment
    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    if not os.path.exists(f"./results/{args.env}"):
        os.makedirs(f"./results/{args.env}")

    # Seed.
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state, _ = env.reset()
    state_dim = state['observation'].shape[0]

    # Environment
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    RL_agent = TD.Agent(state_dim, action_dim, max_action, args)
    name = f"{args.policy}_{args.env}_{args.seed}"

    print("---------------------------------------")
    print(f"Algorithm: {args.policy}, N: {args.N}, Environment: {args.env}, Seed: {args.seed}, "
          f"Device: {RL_agent.device}")
    print("---------------------------------------")

    # Optimize.
    train_online(RL_agent, env, eval_env, args)