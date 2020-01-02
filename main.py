import argparse
import numpy as np
import matplotlib.pyplot as plt
from env import Maze
from algo import InternalModel, dyna_q

if __name__ == '__main__':
    # define argument parser 
    parser = argparse.ArgumentParser()
    parser.add_argument("-runs", "--runs", default=10,
                        help="how many experiments to run")
    parser.add_argument("-episodes", "--episodes", default=50,
                        help="how many episodes to run in each run")
    parser.add_argument("-epislon", "--epislon", default=0.1, 
                        help="probability for exploration")
    parser.add_argument("-gamma", "--gamma", default=0.95, 
                        help="discount factor")
    parser.add_argument("-alpha", "--alpha", default=0.1, 
                        help="learning rate (step size)")
    parser.add_argument("-plan_step", "--plan_step", default=5, 
                        help="planning steps over the learned model")
    args = parser.parse_args()

    # Create an environment
    env = Maze()
    plan_steps = [0, 5, 50]
    steps = np.zeros((len(plan_steps), args.episodes))

    for run in range(args.runs):
        for index, plan_step in zip(range(len(plan_steps)), plan_steps):
            args.planning_steps = plan_step
            print(run, plan_step)
            # initialize Q table
            q_value = np.zeros(env.q_size)
            # generate Dyna-Q model
            model = InternalModel()

            for ep in range(args.episodes):
                steps[index, ep] += dyna_q(args, q_value, model, env)

    # averaging over runs
    steps /= args.runs

    # plotting here
    for i in range(len(plan_steps)):
        plt.plot(steps[i, :], label='%d steps planning ahead' % (plan_steps[i]))
    plt.title('Figure 8.2')
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()
    plt.show()
    plt.close()

