import numpy as np


### The experience replay memory ###
class ReplayBuffer:
  def __init__(self, state_dim, act_dim, size):

    # The following numpy arrays store the following:
    # states, next_states, actions, rewards, done
    self.states_buf = np.zeros([size, state_dim], dtype=np.float32)
    self.next_states_buf = np.zeros([size, state_dim], dtype=np.float32)
    self.acts_buf = np.zeros(size, dtype=np.uint8)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.uint8)

    # The following represent:
    # self.ptr => pointer that tells us where to insert the data
    # self.size => current size of the buffer
    # self.max_size => maximum buffer size
    self.ptr, self.size, self.max_size = 0, 0, size

  '''
  This function stores the state, next_state, action, reward, done
  in their respective buffers at index self.ptr
  '''
  
  def store(self, state_dim, act, rew, next_obs, done):
    self.states_buf[self.ptr] = state_dim
    self.next_states_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done

    # Once we have stored the data, we update the pointer
    # Buffer is a circular array so use modulo to go back to 0 once we reach end of array
    self.ptr = (self.ptr+1) % self.max_size 
    # We also have to update the size of the buffer
    self.size = min(self.size+1, self.max_size)

  '''
  The sample_batch function chooses random indeces from 0 to buffer size
  and then converts the respective data as a dictionary
  '''
  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s=self.states_buf[idxs],
                s2=self.next_states_buf[idxs],
                a=self.acts_buf[idxs],
                r=self.rews_buf[idxs],
                d=self.done_buf[idxs])