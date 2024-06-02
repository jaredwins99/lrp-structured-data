import numpy as np
from torch import nn


def standardize(vars):
    converted = False
    if vars.ndim == 1:
        vars = vars.reshape(vars.size, 1)
        converted = True
    for col in range(vars.shape[1]):
        vars[:,col] = (vars[:,col] - vars[:,col].mean()) / vars[:,col].std()
    if converted:
        vars = vars.flatten()
    return vars


class BasicRegressionNN(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super().__init__()

        # Start with the input layer
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(), nn.Dropout(dropout_rate)]
        
        # Add arbitrary number of hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.extend([
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        # Add the output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Define the feedforward neural network architecture using the layers list
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def train(X_batch, y_batch, model, loss_fn, optimizer, epochs):

    for epoch in range(epochs):

        #for X_batch, y_batch in dataloader:
        
        output = model(X_batch)

        loss = loss_fn(output, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


def lrp(model, features, bias=True):
    # Weights
    W = []
    # Biases
    B = []
    params = model.state_dict()
    for param in params:
        if 'weight' in param:
            W.append(params[param].numpy())
        elif 'bias' in param:
            B.append(params[param].numpy())
    # Number of layers
    L = len(W) 
    # Activations
    A = [features] + [None]*L 

    # Forward pass
    for l in range(L):
        if l == L-1:
            A[l+1] = A[l].dot(W[l].T) + B[l]
        else:
            A[l+1] = relu(A[l].dot(W[l].T) + B[l])
    R = [None]*L + [A[L]]

    ## Top layers
    for l in range(L-1, 0, -1):

        # Extra Rules
        rho_w = rho(W[l].T,l)
        
        if bias:
            rho_b = rho(B[l],l)
        else:
            rho_b = 0

        # Four LRP steps
        z = incr(A[l].dot(rho_w) + rho_b, l)  # step 1: forward pass
        s = R[l+1] / z  # step 2: elementwise division
        c = s.dot(W[l])  # step 3: backward pass
        R[l] = A[l] * c  # step 4: elementwise product

    ## Input Layer (apply the w^2 rule for continuous tabular data)
    w_sqr = W[0].T**2
    z = (w_sqr).sum(axis=0)  # step 1: forward pass
    s = R[1] / z  # step 2: elementwise division
    c = s.dot(w_sqr.T)  # step 3: backward pass
    R[0] = c  # step 4: elementwise product

    # Result
    return R


def relu(x):
    return np.maximum(0, x)


def rho(w,l):  
    gamma = relu(w)
    return w + gamma * [None,0.0,0.0,0.0][l]


def incr(z,l): 

    # Add small quantities to regularize results
    root_mean_square = (z**2).mean()**(1/2)
    numerical_stability = 1e-9
    epsilon = root_mean_square + numerical_stability

    # Depending on the layer, 
    return z + epsilon * [None,0.0,0.0,0.0][l] # None is a reminder to not use the last layer