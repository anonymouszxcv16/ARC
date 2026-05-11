import argparse
import os
import random
import time
import torch
import gym
import procgen

from dqn import Agent
from helpers import PreprocessFrame

def evaluate(args, env, agent, scores, biases, variances, stds, times):
    discounteds = []
    qs = []

    # Mean.
    for idx in range(args.eval_eps):
        state = env.reset()

        if "ALE" in args.env:
            state = state[0]

        state = preprocessor.reset(state)
        state = torch.tensor(state, dtype=torch.float32, device=args.device).unsqueeze(0)

        with torch.no_grad():
            zs_target = agent.fixed_encoder_target.zs(state)
            q = (agent.target_net(state, zs_target).amax(-1).squeeze(-1).clone().detach().to(agent.args.device, dtype=torch.float32))
            qs.append(q)

        step = 0
        rewards = 0

        ep_finished = False

        # Episode.
        while not ep_finished:
            action = agent.select_action(state, q_idx=random.randint(0, args.N - 1))

            if "ALE" in args.env:
                observation, reward, done, trunc, _ = env.step(action.item())
            else:
                observation, reward, done, _ = env.step(action.item())
                trunc = False

            observation = preprocessor.step(observation)
            state = torch.tensor(observation, dtype=torch.float32, device=args.device).unsqueeze(0)

            rewards += reward * agent.args.discount ** step
            step += 1

            ep_finished = done or trunc

        discounteds.append(rewards)

    # Mean
    rewards_avg = torch.tensor(discounteds).mean()
    variance = torch.tensor(discounteds).std()
    qs_avg = torch.tensor(qs).mean()

    bias = torch.tensor(rewards_avg - qs_avg).abs().item()

    # Collect metrics,
    scores.append(rewards_avg)
    variances.append(variance)

    time_tot = (time.time() - time_start) / 60
    times.append(time_tot)

    epsilon = agent.get_epsilon()
    std = (agent.memory.reward[:agent.memory.size].std() / agent.memory.reward[:agent.memory.size].mean()).item()

    # Rewards std
    stds.append(std)
    biases.append(bias)

    # Log score.
    with open(f"./results/RRS/{args.env}/{args.file_name}", "w") as file:
        file.write(f"{scores}\n{times}\n{biases}\n{variances}\n{stds}")

    # Log.
    print(f"Steps: {agent.t:,.1f}\tTimes: {time_tot:,.1f}\tScore: {rewards_avg:,.3f}\t"
          f"Std(R): {std:,.2f}\t"
          f"Bias: {bias:,.2f}\tVariance: {variance:,.2f}\tEpsilon: {epsilon:,.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Algorithm
    parser.add_argument("--policy", default="DQN", type=str)
    parser.add_argument("--N", default=1, type=int)

    # AVD
    parser.add_argument("--avd_alpha", default=0.05, type=float)

    # RRS
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--auto_alpha", default=1, type=int)
    parser.add_argument("--auto_alpha_interval", default=1_000, type=int)
    parser.add_argument("--alpha_max", default=1, type=float)
    parser.add_argument("--alpha_min", default=.5, type=float)

    # Architecture
    parser.add_argument("--hdim", default=256, type=int)
    parser.add_argument("--zs_dim", default=256, type=int)
    parser.add_argument("--encoder_lr", default=3e-4, type=float)

    # Environment
    parser.add_argument("--env", default="FlappyBird-v0", type=str)
    # tetris_gymnasium/Tetris, FlappyBird-v0

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_seeds", default=5, type=int)

    # Evaluation
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--timesteps_before_training", default=25_000, type=int)
    parser.add_argument("--eval_iter", default=int(5e3), type=int)
    parser.add_argument("--eval_eps", default=10, type=int)

    # Experience replay
    parser.add_argument("--replay_size", default=int(1e6), type=int)
    parser.add_argument("--batch_size", default=256, type=int)

    # Hyperparameters
    parser.add_argument("--discount", default=.99, type=float)

    # Epsilon greedy.
    parser.add_argument("--eps_start", default=.9, type=float)
    parser.add_argument("--eps_end", default=.05, type=float)
    parser.add_argument("--eps_decay", default=int(1e6), type=int)

    # Target.
    parser.add_argument("--tau", default=5e-3, type=float)

    # Learning.
    parser.add_argument("--lr", default=3e-5, type=float)

    args = parser.parse_args()

    args.file_name = f"{args.policy}_{args.seed}"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if not os.path.exists(f"{os.getcwd()}/results/RRS/{args.env}"):
        os.makedirs(f"{os.getcwd()}/results/RRS/{args.env}")

    env = gym.make(args.env)
    env_eval = gym.make(args.env)

    preprocessor = PreprocessFrame()
    env.action_space.seed(args.seed)

    state = env.reset()

    if "ALE" in args.env:
        state = state[0]

    state = preprocessor.reset(state)

    args.state_shape = state.shape
    args.action_shape = (1,)
    args.n_actions = env.action_space.n

    print("---------------------------------------")
    print(f"Policy: {args.policy}, N: {args.N}, State Space: {args.state_shape}, Action Space: {env.action_space.n},"
          f" Seed: {args.seed}, Device: {args.device}")
    print("---------------------------------------")

    # Seed.
    torch.manual_seed(args.seed)
    agent = Agent(args, env, preprocessor)

    # Evaluation
    time_start = time.time()

    # Logs.
    scores = []
    times = []
    variances = []
    biases = []
    stds = []

    # Train.
    while agent.t < args.max_timesteps:
        # Initialize the environment and get its state
        state = env.reset()

        if "ALE" in args.env:
            state = state[0]

        # Tetris
        state = agent.preprocessor.reset(state)
        state = torch.tensor(state, dtype=torch.float32, device=args.device).unsqueeze(0)

        ep_finished = False

        # Episode.
        while not ep_finished:
            if agent.t % args.eval_iter == 0:
                # Evaluate RL.
                evaluate(args, env_eval, agent, scores, biases, variances, stds, times)

            agent.t += 1

            action = agent.select_action(state)

            if "ALE" in args.env:
                observation, reward, done, trunc, _ = env.step(action.item())
            else:
                observation, reward, done, _ = env.step(action.item())
                trunc = False

            # Tetris
            observation = agent.preprocessor.step(observation)
            reward = torch.tensor([reward], dtype=torch.float32, device=args.device)

            ep_finished = done or trunc
            done = torch.tensor(done)

            if done:
                next_state = torch.zeros(state.shape, dtype=torch.float32, device=args.device)
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=args.device).unsqueeze(0)

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            if agent.t >= args.timesteps_before_training:
                agent.optimize_model()

            if agent.t == args.max_timesteps:
                break

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()

            # Target.
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * args.tau + target_net_state_dict[key] * (1 - args.tau)

            agent.target_net.load_state_dict(target_net_state_dict)