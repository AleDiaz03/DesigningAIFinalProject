import numpy as np

import torch
import torch.nn as nn

from replayBuffer import ReplayBuffer
from model import MLP, predict, train_one_step


class DQNAgent(object):
  '''
  The constructor takes in the state_size and action_size
  which are the dimensions of the input and output of the neural 
  network respectively
  '''
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = ReplayBuffer(state_size, action_size, size=500)
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = MLP(state_size, action_size)

    # Loss and optimizer
    self.criterion = nn.MSELoss() # We will be using Mean Squared Error
    self.optimizer = torch.optim.Adam(self.model.parameters())

  # Takes state, next_state, action, reward and done and stores it in the buffer
  def update_replay_memory(self, state, action, reward, next_state, done):
    self.memory.store(state, action, reward, next_state, done)

  # This takes the state and uses epsilon greedy to come up with the action
  def act(self, state):
    # We get a random number
    # If it is less than epsilon, perform a random action
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)
    # If greater than epsilon, get the Q values for all actions and perform best action
    act_values = predict(self.model, state)
    return np.argmax(act_values[0])  # returns action

  '''
  Most important function in this class
  '''
  def replay(self, batch_size=32):
    # first check if replay buffer contains enough data
    # if not, we can't grab a full batch so return
    if self.memory.size < batch_size:
      return 

    # sample a batch of data from the replay memory
    minibatch = self.memory.sample_batch(batch_size)
    # sample_batch returns a dictionary so we retrieve the data
    #   using the corresponding keys
    states = minibatch['s']
    actions = minibatch['a']
    rewards = minibatch['r']
    next_states = minibatch['s2']
    done = minibatch['d']

    # Calculate the target: Q(s',a)
    target = rewards +  self.gamma * np.amax(predict(self.model, next_states), axis=1)

    '''
    - The target is a 1D array of size batch size
    - Model prediction is a 2D array of size batch_size*n_action
    - So target and predictions don't have the same shape
    - This is necessary to calulate loss
    - For each sample, we must have a target for each action 
      even if that action is not the one that the agent took
    - With the PyTorch API, it is simplest to have the target be the 
      same shape as the predictions. However, we only need to update the network for the actions
      which were actually taken. We can accomplish this by setting the target to be equal to
      the prediction for all values.
    - Then, only change the targets for the actions taken. Q(s,a)
    - As for the actions not taken target = prediction, the loss for these
      will be 0 and so won't affect our model
    '''

    target_full = predict(self.model, states)
    target_full[np.arange(batch_size), actions] = target

    # Run one training step
    train_one_step(self.model, self.criterion, self.optimizer, states, target_full)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay


  def load(self, name):
    self.model.load_weights(name)


  def save(self, name):
    self.model.save_weights(name)
