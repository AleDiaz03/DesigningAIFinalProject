import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  def __init__(self, n_inputs, n_action, n_hidden_layers=1, hidden_dim=32):
    super(MLP, self).__init__()

    M = n_inputs
    self.layers = []

    # This loop adds the hidden layers to our model
    for _ in range(n_hidden_layers):
      # M is the input to the layer and hidden_dim is the output
      layer = nn.Linear(M, hidden_dim) 
      M = hidden_dim
      self.layers.append(layer)
      # We use ReLU as the activation function
      self.layers.append(nn.ReLU())

    # final layer (output layer)
    # Remember the otuput of the model must be of size n_action
    self.layers.append(nn.Linear(M, n_action))
    # Create a Sequential object with all these layers
    self.layers = nn.Sequential(*self.layers)

  # Forward just passes the data through our model's layers
  def forward(self, X):
    return self.layers(X)
  
  # Function to save the model
  def save_weights(self, path):
    torch.save(self.state_dict(), path)

  # Function to load the model
  def load_weights(self, path):
    self.load_state_dict(torch.load(path))


############################################################
# FUNCTION TO MAKE PREDICTION
############################################################
def predict(model, np_states):
  # no_grad disables gradient tracking
  # we use this as we are not training the model just making a prediction
  with torch.no_grad():
    # Convert the states to float32 and then to a torch tensor
    inputs = torch.from_numpy(np_states.astype(np.float32))
    # Get the prediction
    output = model(inputs)
    # Return prediction as numpy array
    return output.numpy()
  

############################################################
# FUNCTION TO TRAIN
############################################################
'''
This runs a single step of gradient descent
'''
def train_one_step(model, criterion, optimizer, inputs, targets):
  # convert to tensors
  inputs = torch.from_numpy(inputs.astype(np.float32))
  targets = torch.from_numpy(targets.astype(np.float32))

  # zero the parameter gradients
  optimizer.zero_grad()

  # Forward pass
  outputs = model(inputs)
  loss = criterion(outputs, targets)
        
  # Backward and optimize
  loss.backward()
  optimizer.step()
