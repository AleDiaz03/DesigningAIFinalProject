import numpy as np
import argparse
import pickle
from datetime import datetime


from helper import find_or_make_dir, get_data, get_scaler
from environment import MultiStockEnv
from agent import DQNAgent



def play_one_episode(agent, env, is_train):
  # note: after transforming states are already 1xD

  # Reset the environment and scale the state
  state = env.reset()
  state = scaler.transform([state])
  done = False

  # This loop is where we take all steps of the episode
  while not done:
    # Get the action, peform it and get the new state data
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    next_state = scaler.transform([next_state])

    # If we are training the model, update memory and run one step of gradient descent
    if is_train == 'train':
      agent.update_replay_memory(state, action, reward, next_state, done)
      agent.replay(batch_size)
    
    # Set state to next_state for next iteration of the loop
    state = next_state

  # Once the episode is over, we return the final value of the portfolio
  return info['cur_val']


'''
THIS IS THE MAIN SECTION WHERE EVERYTHING COMES TOGETHER
'''

if __name__ == '__main__':
  # config vars
  models_folder = 'rl_trader_models'
  rewards_folder = 'rl_trader_rewards'
  num_episodes = 2000
  batch_size = 32
  initial_investment = 20000

  # We instantiate an argument parser so we can run the script with commandline arguments
  # We will have 1 argument (mode) and we can pass train or test
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  args = parser.parse_args()

  find_or_make_dir(models_folder)
  find_or_make_dir(rewards_folder)

  # Get time series data
  data = get_data()
  n_timesteps, n_stocks = data.shape

  # Split data into train and test
  n_train = n_timesteps // 2

  train_data = data[:n_train]
  test_data = data[n_train:]

  # Instantiate the environment, agent and scaler
  env = MultiStockEnv(train_data, initial_investment)
  state_size = env.state_dim
  action_size = len(env.action_space)
  agent = DQNAgent(state_size, action_size)
  scaler = get_scaler(env)

  # store the final value of the portfolio for each episode
  portfolio_value = []


  # NOTE: if we are in test mode, we will overwrite some things we just instantiated
  if args.mode == 'test':
    # load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # Remake the env with test data
    env = MultiStockEnv(test_data, initial_investment)

    # Make sure epsilon is not 1! (pure exploration)
    # No need to run multiple episodes if epsilon = 0, it's deterministic
    agent.epsilon = 0.01

    # load trained weights
    agent.load(f'{models_folder}/dqn.ckpt')



  # Play the game num_episodes times
  for e in range(num_episodes):
    # Grab the time to know the duration of each loop iteration
    t0 = datetime.now()
    val = play_one_episode(agent, env, args.mode)
    dt = datetime.now() - t0
    print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
    portfolio_value.append(val) # append episode end portfolio value

  # Save the weights when we are done
  if args.mode == 'train':
    # Save the DQN
    agent.save(f'{models_folder}/dqn.ckpt')

    # save the scaler
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)


  # save portfolio value for each episode
  np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)