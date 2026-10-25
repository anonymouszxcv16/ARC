import argparse
import os
import random
import time
import torch
import gym
import procgen

from dqn import Agent
from helpers import PreprocessFrame

def evaluate(args, env, agent,
             scores, biases, variances,

             r_avgs,
             mc_avgs,

             alphas,

             times):
    discounteds = []
    qs = []

    # Mean.
    for idx in range(args.eval_eps):
        state = env.reset()

        state = preprocessor.reset(state)
        state = torch.tensor(state, dtype=torch.float32, device=args.device).unsqueeze(0)

        with torch.no_grad():
            zs_target = agent.fixed_encoder_target.zs(state)
            q = (agent.target_net(state, zs_target).amax(-1).squeeze(-1).clone().detach().to(agent.args.device, dtype=torch.float32))
            qs.append(q)

        step = 0
        rewards = 0

        ep_finished = False

        # old: done = False

        # Episode.
        while not ep_finished:
            action = agent.select_action(state, q_idx=random.randint(0, args.N - 1))

            observation, reward, done, _ = env.step(action.item())

            observation = preprocessor.step(observation)
            state = torch.tensor(observation, dtype=torch.float32, device=args.device).unsqueeze(0)

            rewards += reward * agent.args.discount ** step
            step += 1

            ep_finished = done

        discounteds.append(rewards)

    # Mean
    rewards_avg = torch.tensor(discounteds).mean()
    variance = torch.tensor(discounteds).std()

    bias = (torch.tensor(qs) - torch.tensor(discounteds)).mean()

    time_tot = (time.time() - time_start) / 60
    times.append(time_tot)

    # R, MC
    r_avg = agent.memory.reward[:agent.memory.size].mean().item()
    mc_avg = agent.memory.mc_score[:agent.memory.size].mean().item()

    scores.append(rewards_avg.item())
    variances.append(variance.item())

    # Rewards std
    r_avgs.append(r_avg)
    mc_avgs.append(mc_avg)

    biases.append(bias.item())

    alphas.append(agent.args.alpha)

    # Log score.
    with open(f"./results/RRS/{args.env}/{args.file_name}", "w") as file:
        file.write(f"{scores}\n{times}\n{biases}\n{variances}\n{r_avgs}\n{mc_avgs}\n{alphas}")

    # Log.
    print(f"Steps: {agent.t:,.1f}\tTimes: {time_tot:,.1f}\tScore: {rewards_avg:,.3f}\t"          
          f"Avg(R): {r_avg:,.2f}\tAvg[MC]: {mc_avg:,.2f}\t"
          f"Bias: {bias:,.2f}\tVariance: {variance:,.2f}\t"
          f"Alpha: {agent.args.alpha:,.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Algorithm
    parser.add_argument("--policy", default="DQN", type=str)
    parser.add_argument("--N", default=1, type=int)

    # AVD
    parser.add_argument("--avd_alpha", default=0.05, type=float)

    # RRS
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--auto_alpha", default=0, type=int)
    parser.add_argument("--auto_alpha_interval", default=1_000, type=int)
    parser.add_argument("--alpha_max", default=1, type=float)
    parser.add_argument("--alpha_min", default=.5, type=float)

    parser.add_argument("--eps", default=0.2, type=float)

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
    parser.add_argument("--timesteps_before_training", default=1_000, type=int)
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
    # old: env_eval = gym.make(args.env, num_levels=0)
    env_eval = gym.make(args.env)

    preprocessor = PreprocessFrame(grayscale=False)
    env.action_space.seed(args.seed)

    state = env.reset()

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

    count_mc_shift_tots = []
    count_r_shift_tots = []

    r_avgs = []
    mc_avgs = []
    mc_srs_avgs = []
    alphas = []

    ep_total_reward, ep_timesteps = 0, 0
    agent.t0 = 0

    # Train.
    while agent.t < args.max_timesteps:
        # Initialize the environment and get its state
        state = env.reset()

        # Tetris
        state = agent.preprocessor.reset(state)
        state = torch.tensor(state, dtype=torch.float32, device=args.device).unsqueeze(0)

        ep_finished = False

        # Episode.
        while not ep_finished:
            if agent.t % args.eval_iter == 0:
                # Evaluate RL.
                evaluate(args, env_eval, agent,
                         scores, biases, variances,

                         r_avgs,
                         mc_avgs,

                         alphas,

                         times)

            agent.t += 1

            action = agent.select_action(state)
            observation, reward, done, _ = env.step(action.item())

            # Tetris
            observation = agent.preprocessor.step(observation)
            reward = torch.tensor([reward], dtype=torch.float32, device=args.device)

            ep_finished = done
            done = torch.tensor(done)

            ep_total_reward += reward
            ep_timesteps += 1

            if done:
                next_state = torch.zeros(state.shape, dtype=torch.float32, device=args.device)
                ep_total_reward, ep_timesteps = 0, 0

                # Episode length
                T = agent.t - agent.t0
                mc_scores = torch.zeros((T, 1)).to(args.device)
                mc_scores_srs = torch.zeros((T, 1)).to(args.device)

                # Gt
                for step in range(T):
                    # t
                    for t in range(step, T):
                        # r_t
                        r_t = agent.memory.reward[agent.t - agent.t0 + t]
                        r_t_srs = agent.memory.reward[agent.t - agent.t0 + t]

                        if "RRS" in args.policy or "NRS" in args.policy:
                            # Q-targets.
                            rewards_mean, rewards_max = agent.memory.reward[:agent.memory.size].mean(), agent.memory.reward[:agent.memory.size].max()
                            r_t_srs = ((r_t_srs - rewards_mean) / (rewards_max if rewards_max > 0 else 1))

                            if "RRS" in args.policy:
                                r_t_srs = (1 / args.alpha) * 1 / (1 + (-args.alpha * r_t_srs).exp()) * (1 + (args.alpha * r_t_srs).exp()).log()

                        mc_scores[step] += r_t * args.discount ** (t - step)
                        mc_scores_srs[step] += r_t_srs * args.discount ** (t - step)

                if agent.t0 % args.replay_size > agent.t % args.replay_size:
                    idxs = list(range(agent.t0 % args.replay_size, args.replay_size)) + list(
                        range(agent.t % args.replay_size))
                else:
                    idxs = range(agent.t0 % args.replay_size, agent.t % args.replay_size)

                agent.memory.mc_score[idxs] = mc_scores
                agent.memory.mc_score_srs[idxs] = mc_scores_srs
                agent.t0 = agent.t

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