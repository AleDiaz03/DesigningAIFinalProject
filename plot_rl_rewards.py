import matplotlib.pyplot as plt
import numpy as np
import argparse

'''
This script loads the rewards (can switch to train or test)
Prints avg, min and max reward
Plots histogram of rewards over each episode
'''

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='eigher "train" or "test"')
args = parser.parse_args()
a = np.load(f'rl_trader_rewards/{args.mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

plt.hist(a, bins=20)
plt.title(args.mode)
plt.show()