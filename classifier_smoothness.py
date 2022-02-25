import numpy as np
import torch
from torch import linalg as LA

"""
Various ideas to quantify the smoothness (across the boundary) 
of the decision boundaries in 2D binary classification problems. 
The network is expected to output likelihood values in (0,1).
"""

def max_class_var_2d(x_min_max, y_min_max, model, dx=0.001, dy=0.001, n=100):
    """ 
    Spans a grid over the classification domain and computes the differences 
    of network outputs at adjacent grid points. Returns the average of the n 
    largest such differences for x- and y-direction.
    """
    
    # create grid with spacing dx
    x_min, x_max = x_min_max[0], x_min_max[1]
    y_min, y_max = y_min_max[0], y_min_max[1]
    x_arange = np.arange(x_min, x_max, dx).reshape(-1,1)
    y_arange = np.arange(y_min, y_max, dy).reshape(-1,1)
    Nx = len(x_arange)
    Ny = len(y_arange)
    x_grid = np.hstack( (x_arange, np.zeros(Nx).reshape(-1,1)) )
    y_grid = np.hstack( (np.zeros(Ny).reshape(-1,1), y_arange) )
    
    # feed grid points into network
    x_grid_prob = model.forward(torch.tensor(x_grid))
    y_grid_prob = model.forward(torch.tensor(y_grid))
    
    # turn to np arrays
    x_grid_prob = x_grid_prob.detach().numpy().reshape(Nx)
    y_grid_prob = y_grid_prob.detach().numpy().reshape(Ny)
    
    # calculate variances
    x_var = np.array([np.abs(x_grid_prob[i+1]-x_grid_prob[i]) for i in range(0, Nx-1)])
    y_var = np.array([np.abs(y_grid_prob[i+1]-y_grid_prob[i]) for i in range(0, Ny-1)])
    
    # sort them by size
    x_var_sort = x_var[np.argsort(x_var)]
    y_var_sort = y_var[np.argsort(y_var)]
    
    # take average of n largest variances
    return (np.mean(x_var_sort[-n:]), np.mean(y_var_sort[-n:]) )
    
    
def integral_over_class_2d(x_min_max, y_min_max, model, dx=0.001, dy=0.001):
    """
    Spans a grid over the classification domain and computes the sum of 
    probabilities to belong to class 1, as given by the network, 
    along the x- and y-directions (i.e. one value for each grid line). 
    Returns average for both directions.
    
    """
    
    # create grid with spacing dx
    x_min, x_max = x_min_max[0], x_min_max[1]
    y_min, y_max = y_min_max[0], y_min_max[1]
    x_arange = np.arange(x_min, x_max, dx).reshape(-1,1)
    y_arange = np.arange(y_min, y_max, dy).reshape(-1,1)
    Nx = len(x_arange)
    Ny = len(y_arange)
    x_grid = np.hstack( (x_arange, np.zeros(Nx).reshape(-1,1)) )
    y_grid = np.hstack( (np.zeros(Ny).reshape(-1,1), y_arange) )
    
    # feed grid points into network
    x_grid_prob = model.forward(torch.tensor(x_grid))
    y_grid_prob = model.forward(torch.tensor(y_grid))
    
    # turn to np arrays
    x_grid_prob = x_grid_prob.detach().numpy().reshape(Nx)
    y_grid_prob = y_grid_prob.detach().numpy().reshape(Ny)
    
    # convert probabilities p < 0.5 to 1-p
    idx = x_grid_prob < 0.5
    x_grid_prob[idx] = 1-x_grid_prob[idx]
    idx = y_grid_prob < 0.5
    y_grid_prob[idx] = 1-y_grid_prob[idx]
    
    # sum over probabilities
    return(np.sum(x_grid_prob)/Nx, np.sum(y_grid_prob)/Ny)


def gradients_decision_boundary(x_min_max, y_min_max, model, dx=0.01, dy=0.01, eps=0.02):
    """
    Spans a grid over the classification domain and computes the average absolute
    gradient of the network output close to the decision boundaries, i.e. at points x
    where the network output f(x) is in [0.5-eps, 0.5+eps]. Computationally expensive,
    as PyTorch does not support vectorized gradient evaluations.
    """
    
    # create grid with spacing dx
    x_min, x_max = x_min_max[0], x_min_max[1]
    y_min, y_max = y_min_max[0], y_min_max[1]
    x_arange = np.arange(x_min, x_max, dx).reshape(-1,1)
    y_arange = np.arange(y_min, y_max, dy).reshape(-1,1)
    Nx = len(x_arange)
    Ny = len(y_arange)
    x_grid = np.hstack( (x_arange, np.zeros(Nx).reshape(-1,1)) )
    y_grid = np.hstack( (np.zeros(Ny).reshape(-1,1), y_arange) )
    grid = np.vstack((x_grid, y_grid))

    # turn to tensors with requires_grad
    grid = torch.tensor(grid)
    grid.requires_grad = True

    
    # feed grid points into network
    grid_prob = model.forward(grid)
        
    # find points with p ~ 0.5
    idx = np.nonzero( np.abs(grid_prob.detach().squeeze().numpy() - 0.5) < eps )
    print(len(idx[0]))
    
    grad_sizes = []
    model.zero_grad()
    for i in idx[0]:
        grid_prob[i].backward(retain_graph = True)
        grad_sizes.append(LA.norm(grid.grad[i]))
        model.zero_grad()
        
    return np.mean(np.array(grad_sizes))