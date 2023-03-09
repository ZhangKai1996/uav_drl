import time

import numpy as np
import torch as th
from threading import Thread

from algo import Trainer
from env import AirSimDroneEnv
from parameters import ip_address

root = "C:\\Kai Zhang\\Documents\\AirSim\\"
setting_path = root + "settings.json"


def input_format(keyword, y_desc='Yes', n_desc='No'):
    while True:
        inputs = input("{} (Yes/yes/y/1):".format(keyword))
        if inputs in ['Yes', 'yes', 'y', '1']:
            print('\t--> YES:', y_desc)
            print('\t     NO:', n_desc, '\n')
            return True
        elif inputs in ['No', 'no', 'n', '0']:
            print('\t    YES:', y_desc)
            print('\t-->  NO:', n_desc, '\n')
            return False
        else:
            print('Please input again')


def make_exp_id(args):
    return 'exp_{}_{}_{}_{}_{}_{}_{}'.format(args.exp_name, args.seed,
                                             args.a_lr, args.c_lr, args.batch_size, args.gamma,
                                             time.time())


def train(args):
    # Seed
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    # Create environment
    env = AirSimDroneEnv(ip_address=ip_address,
                         image_shape=(3, 180, 292),
                         step_length=1.0)
    # Create MARL trainer
    trainer = Trainer(dim_obs=env.image_shape,
                      dim_act=env.action_space.shape[0],
                      args=args,
                      folder=make_exp_id(args_))
    # Load previous param
    if args.load_dir is not None:
        trainer.load_model(load_path=args.load_dir)

    # Start iterations
    print('Iteration start...')
    step, episode, reward_step, reward_epi, steps = 0, 0, [], [], []
    start = time.time()
    thread = None
    while True:
        episode += 1
        obs = env.reset()
        reward_per_epi = 0.0
        for i in range(args.max_episode_len):
            act = trainer.act(obs, step >= args.learning_start)
            next_obs, rew, done, _ = env.step(act, render=args.render)

            step += 1
            reward_step.append(rew)
            reward_per_epi += rew
            trainer.add_experience(obs, act, next_obs, rew, float(done))
            obs = next_obs

            end = time.time()
            print("{:>3d}, {:>5d}, {:>5d}".format(i, step, episode),
                  ["{:>+.2f}".format(a) for a in act], "{:>+7.3f}".format(rew),
                  "{:>5.2f}".format(end - start))
            start = end

            if step > args.learning_start:
                # trainer.update(step)
                if thread is None:
                    thread = Thread(target=trainer.update)
                    thread.start()
                elif not thread.is_alive():
                    thread = None

            if done or i >= args.max_episode_len - 1:
                reward_epi.append(reward_per_epi)
                steps.append(i)
                break

        if episode % args.save_rate == 0:
            trainer.save_model()

            # Mean reward for each agent (step)
            trainer.scalar("Reward_step", np.mean(reward_step), episode)
            # Mean reward for each agent (episode)
            trainer.scalar("Reward_epi", np.mean(reward_epi), episode)
            trainer.scalar("Average_step", np.mean(steps), episode)
            reward_step, reward_epi, steps = [], [], []

        if episode >= args.num_episodes:
            break
    # End environment
    env.close()
    trainer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
    # Environment
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=1000, help="number of episodes")
    parser.add_argument('--memory-length', default=int(1e6), type=int, help='number of experience replay pool')
    parser.add_argument("--learning-start", type=int, default=50, help="start updating after this number of step")
    parser.add_argument("--good-policy", type=str, default="algo", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="algo", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--a-lr", type=float, default=1e-4, help="learning rate for Actor Adam optimizer")
    parser.add_argument("--c-lr", type=float, default=1e-3, help="learning rate for Critic Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument('--tau', default=0.001, type=float, help='rate of soft update')
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='train', help="name of the experiment")
    parser.add_argument("--seed", type=int, default=1111, help="name of the experiment")
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument("--save-rate", type=int, default=10,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default=None,
                        help="directory in which training state and model are loaded")

    args_ = parser.parse_args()
    if input_format(keyword='The Unreal client has been opened',
                    y_desc='Execute the train function',
                    n_desc='Not ready yet!'):
        train(args=args_)
