import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


############################################################
# FUNCTION TO GET DATA
############################################################

# Let's use AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
# We are only using the closing price for each stock
def get_data():
  # returns a T x 3 list of stock prices
  # each row is a different stock
  # 0 = AAPL
  # 1 = MSI
  # 2 = SBUX
  df = pd.read_csv('aapl_msi_sbux.csv')
  return df.values


############################################################
# FUNCITON TO CREATE OUR SCALER
############################################################

# Takes in an environment object to fit our scaler
def get_scaler(env):
  # return scikit-learn scaler object to scale the states
  # Note: you could also populate the replay buffer here

  # In order to have the right parameters for our scaler we must have some data
  # So we play random episodes (random actions) to get some data and fit the scaler
  states = []
  for _ in range(env.n_step):
    action = np.random.choice(env.action_space)
    state, reward, done, info = env.step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler


############################################################
# FUNCTION TO MAKE A DIRECTORY
############################################################

'''
Checks whether a particular directory exists and if not creates
a new one.
'''
def find_or_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)