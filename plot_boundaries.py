import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy


def plt_boundary(Xtrain, model, batch_features=[], plot_title=" "):
    """
    Method to plot the class boundaries of a (trained) model on 
    2D binary classification problem (features and labels stored in Xtrain). 
    The color density within classes corresponds to the networks given probability 
    for the area to belong to the class. Thus, the model needs to give an output in (0,1).
    With batch_features, a single batch of data points can also be plotted. 
    """
    
    ### create grid with N points in each dimension
    x_min, x_max = Xtrain.numpy()[:,0].min()-0.1, Xtrain.numpy()[:,0].max()+0.1
    y_min, y_max = Xtrain.numpy()[:,1].min()-0.1, Xtrain.numpy()[:,1].max()+0.1
    N = 500
    XX, YY = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/N),
                         np.arange(y_min, y_max, (y_max-y_min)/N))
   
    ### feed grid points into network
    grid_data = np.hstack((XX.ravel().reshape(-1,1), 
                          YY.ravel().reshape(-1,1)))
    grid_data_t = torch.tensor(grid_data)
    grid_prob = model.forward(grid_data_t)
    
    ### plot class boundaries + data set + batch    
    classes = grid_prob >= 0.5

    class0 = np.copy(classes)
    class0 = class0==0
    class0 = class0 * (1-grid_prob.detach().numpy())
    class0 = np.ma.masked_where(class0==0, class0)
    class1 = np.copy(classes)
    class1 = class1==1
    class1 = class1 * grid_prob.detach().numpy()
    class1 = np.ma.masked_where(class1==0, class1)

    
    plt.figure(figsize=(12,8))
    color0 = copy.copy(mpl.cm.get_cmap("Greens"))
    color0.set_bad(color="white")
    color1 = copy.copy(mpl.cm.get_cmap("Oranges"))
    color1.set_bad(color="white")

    plt.contourf(XX, YY, class0.reshape(-1,XX.shape[1]), cmap=color0, alpha=0.75)
    plt.contourf(XX, YY, class1.reshape(-1,XX.shape[1]), cmap=color1, alpha=0.75)
    N=int(len(Xtrain)/2)
    plt.scatter(Xtrain.numpy()[0:N-1,0], Xtrain.numpy()[0:N-1,1], s=3, label="Class 1", color="g")
    plt.scatter(Xtrain.numpy()[N:2*N-1,0], Xtrain.numpy()[N:2*N-1,1], s=3, label="Class 2", color="orange")
    #plt.scatter(batch_features[:,0], batch_features[:,1], color="red", label="Batch")                          # UNCOMMENT to plot batch
    plt.legend()
    plt.title(plot_title)
    #plt.savefig(plot_title+".png", bbox_inches='tight')
    plt.show()