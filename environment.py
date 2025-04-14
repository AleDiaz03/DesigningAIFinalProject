import numpy as np
import itertools

class MultiStockEnv:
  """
  A 3-stock trading environment.
  State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
    - cash owned (can be used to purchase more stocks)
  Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
  """
  def __init__(self, data, initial_investment=20000):
    # data
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    # instance attributes
    self.initial_investment = initial_investment
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    # We are going to have 3 stocks so len(action_space) = 3^3 = 27
    # so action space will be a np array with integers from 0 to 26
    self.action_space = np.arange(3**self.n_stock)

    '''
    action permutations returns a nested list with elements like:
    # [0,0,0] => sell all your stocks
    # [0,0,1] => sell first two stocks and hold third stock
    0 = sell, 1 = hold, 2 = buy
    so itertools.product returns all possible combinations of 0, 1 and 2
    and repeat = n_stocks makes it so that each combination includes the three numbers
    as we are considering 3 stocks
    If n_stocks was to be 2 it would return [0, 0], [0, 1], [1, 0]...
    '''
    self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    # Length of the state array is 2(n_stock) + 1
    # As we need n_shares we own for each stock, price of each stock and total cash to invest
    self.state_dim = self.n_stock * 2 + 1

    # This returns the initial state
    self.reset()


  def reset(self):
    # This means point to the first day of stock prices in our dataset
    self.cur_step = 0
    # Set stock_owned to an array of 0s as in the beginning we don't have any
    self.stock_owned = np.zeros(self.n_stock)
    # Set the stock prices at the current day
    self.stock_price = self.stock_price_history[self.cur_step]
    self.cash_in_hand = self.initial_investment
    return self._get_state()

    '''
    This function performs an action and returns the next state and reward
    '''
  def step(self, action):

    # Check if the action we received exists in our action space
    assert action in self.action_space

    # get current value of our portfolio before performing the action
    prev_val = self._get_val()

    # update price, i.e. go to the next day
    self.cur_step += 1
    self.stock_price = self.stock_price_history[self.cur_step]

    # perform the trade
    self._trade(action)

    # get the new value of portfolio after taking the action
    cur_val = self._get_val()

    # reward is the increase in porfolio value
    reward = cur_val - prev_val

    # done if we have run out of data
    done = self.cur_step == self.n_step - 1

    # store the current value of the portfolio here
    info = {'cur_val': cur_val}

    # conform to the Gym API
    return self._get_state(), reward, done, info


  '''
  This function returns the formatted state array
  '''
  def _get_state(self):
    # Create an empty array 
    state = np.empty(self.state_dim)

    # Add the components of the state
    state[:self.n_stock] = self.stock_owned
    state[self.n_stock:2*self.n_stock] = self.stock_price
    state[-1] = self.cash_in_hand

    return state
    

  '''
  This function returns the value of our portfolio
  So for each stock is the number of shares we own * share price 
  and we add also the cash we have
  '''
  def _get_val(self):
    return self.stock_owned.dot(self.stock_price) + self.cash_in_hand


  '''
  This function performs the action we have selected
  '''
  def _trade(self, action):
    # index the action we want to perform
    # 0 = sell
    # 1 = hold
    # 2 = buy
    # e.g. [2,1,0] means:
    # buy first stock
    # hold second stock
    # sell third stock

    # Retrieve the action vector from our action list
    # Remember that the action we recieve is the index in which the action is 
    action_vec = self.action_list[action]

    # determine which stocks to buy or sell
    sell_index = [] # stores index of stocks we want to sell
    buy_index = [] # stores index of stocks we want to buy
    # Loop through our action vector
    for i, a in enumerate(action_vec):
      # If value at index i is 0, it means we want to sell the stock in index i
      if a == 0:
        sell_index.append(i)
      # If value at index i is 2, it means we want to buy the stock in index i
      elif a == 2:
        buy_index.append(i)


    '''
    sell any stocks we want to sell
    then buy any stocks we want to buy
    '''
    if sell_index:
      # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.stock_owned[i] = 0
    if buy_index:
      # NOTE: when buying, we will loop through each stock we want to buy,
      #       and buy one share at a time until we run out of cash
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned[i] += 1 # buy one share
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False