import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import time

def plot_loss_min_1D(model, batches_train, N_grid=1000, seed=0):
    """
    1-dimensional projection of loss surface of SHLP network (model)
    along random direction in parameter space.
    """
    
    print("Evaluating loss along 1 direction in parameter space...")
    torch.manual_seed(seed)
    
    alpha0 = 1
    
    
    dummy_model = copy.deepcopy(model)                           # make a model copy to evaluate loss surface
    
    ### create random direction in parameter space    
    params = list(model.parameters())                            # list of parameters
    N = len(params)                                              # no. of network parameters
    n_list = []                                                  # list storing sizes of parameters
    for i in range(0, N):
        n_list.append(params[i].numel())
    n = np.sum(n_list)                                           # total number of elements
    delta1 = torch.normal(torch.zeros(n), std=torch.ones(n))     # random direction
    delta1 /= torch.norm(delta1)                                 # normalize
    
    ### split direction vector into subparts fitting parameters' shapes
    delta1_sub = [ delta1[0:n_list[0]].view(params[0].shape).clone() ]
    for i in range(1, N):
        indx_start = np.sum(n_list[0:i])
        indx_end = indx_start + n_list[i]
        delta1_sub.append( delta1[indx_start : indx_end].view(params[i].shape).clone() )
    
    ### store loss values along random direction
    loss_vals = []
    alpha_vals = np.linspace(-alpha0, alpha0, N_grid)
    dummy_params = [0]*N
    with torch.no_grad():
        for alpha in alpha_vals:                                    # perturb network parameters along random direction
            dummy_state_dict = dummy_model.state_dict()
            dummy_params[0] = params[0] + alpha*delta1_sub[0]
            dummy_state_dict["lin1.weight"] = dummy_params[0]
            dummy_params[1] = params[1] + alpha*delta1_sub[1]
            dummy_state_dict["lin1.bias"] = dummy_params[1]
            dummy_params[2] = params[2] + alpha*delta1_sub[2]
            dummy_state_dict["lin2.weight"] = dummy_params[2]
            dummy_params[3] = params[3] + alpha*delta1_sub[3]
            dummy_state_dict["lin2.bias"] = dummy_params[3]
            
            dummy_model.load_state_dict(dummy_state_dict)           # load perturbed parameters into dummy model
            loss_vals.append(dummy_model.test(batches_train)[0])    # evaluate model
            
    ### plot results
    plt.subplots()
    plt.plot(alpha_vals, loss_vals)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("loss")
    plt.title(r"$L(\theta^{*} + \alpha  \theta_r)$ for random $\theta_r$, spirals")     # later: pass title as argument
    plt.show()
    
    

def plot_loss_min_2D(model, batches_train, criterion, N_grid=1000, seed=0):
    """
    Plots projection of loss surface of (trained) model on
    two random directions in parameter space.
    """
    
    print("Evaluating loss along 2 directions in parameter space...")
    torch.manual_seed(seed)
    start_time = time.time()
    
    alpha0 = 10
    
    ### create random directions in parameter space    
    params = list(model.parameters())                                   # list of parameters
    N = len(params)                                                     # no. of network parameters
    n_list = []                                                         # list storing sizes of parameters
    for i in range(0, N):
        n_list.append(params[i].numel())
    n = np.sum(n_list)                                                  # total number of elements
    delta1 = torch.normal(torch.zeros(n), std=torch.ones(n))            # random direction
    delta1 /= torch.norm(delta1)                                        # normalize
    delta2 = torch.normal(torch.zeros(n), std=torch.ones(n))
    delta2 /= torch.norm(delta2)
    
    ### split direction vector into subparts fitting parameters' shapes
    delta1_sub = [ delta1[0:n_list[0]].view(params[0].shape).clone() ]
    delta2_sub = [ delta2[0:n_list[0]].view(params[0].shape).clone() ]
    for i in range(1, N):
        indx_start = np.sum(n_list[0:i])
        indx_end = indx_start + n_list[i]
        delta1_sub.append( delta1[indx_start : indx_end].view(params[i].shape).clone() )
        delta2_sub.append( delta2[indx_start : indx_end].view(params[i].shape).clone() )
        
    
    ### store loss values along random direction
    loss_vals = np.zeros((N_grid,N_grid))
    alpha_vals1 = np.linspace(-alpha0, alpha0, N_grid)
    alpha_vals2 = np.linspace(-alpha0, alpha0, N_grid)
    dummy_model = copy.deepcopy(model)                                   # make a model copy to evaluate loss surface
    dummy_state_dict = dummy_model.state_dict()
    param_keys = list(dummy_state_dict.keys())
    dummy_params = [0]*N
    with torch.no_grad():
        for a1 in range(0,len(alpha_vals1)):
            alpha1 = alpha_vals1[a1]
            for a2 in range(0,len(alpha_vals2)):
                alpha2 = alpha_vals2[a2]
                ### perturb network parameters along random direction
                for i in range(0,N):
                    dummy_params[i] = params[i] + alpha1*delta1_sub[i] + alpha2*delta2_sub[i]
                    dummy_state_dict[param_keys[i]] = dummy_params[i]
          
                dummy_model.load_state_dict(dummy_state_dict)                        # load perturbed parameters into dummy model
                loss_vals[a1][a2] = dummy_model.test(batches_train, criterion)[0]    # evaluate model
            

    print("...took {} minutes".format( (time.time()-start_time)/60 ) )    

    ### plot results
    alpha_vals2, alpha_vals1 = np.meshgrid(alpha_vals2, alpha_vals1)  # CAREFUL WITH ORDER!
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(alpha_vals2, alpha_vals1, loss_vals)
    ax.set_xlabel("\n" + r"$\alpha_2$")
    ax.set_ylabel("\n" + r"$\alpha_1$")
    ax.set_zlabel("\n loss")
    plt.title(r"$L(\theta^{*} + \alpha_1 \theta_{{r1}} + \alpha_2 \theta_{{r2}})$," 
              +"Spirals, AdLaLa, 2 hidden layers")                                    # later: pass title as argument
    plt.show()
    
    
    
def plot_loss_min_2D_molec(model, batches_train, criterion, N_grid=1000, seed=0):
    """
    The same as plot_loss_min_2D but expects model Net_molec
    (see file "torch_network_architectures.py"). 
    """ 
    
    print("Evaluating loss along 2 directions in parameter space...")
    torch.manual_seed(seed)
    start_time = time.time()
    
    alpha0 = 100
    
    ### create random direction in parameter space    
    params = list(model.parameters())                                           # list of parameters
    N = len(params)                                                             # no. of network parameters
    n_list = []                                                                 # list storing sizes of parameters
    for i in range(0, N):
        n_list.append(params[i].numel())
    n = np.sum(n_list)                                                          # total number of elements
    delta1 = torch.normal(torch.zeros(n), std=torch.ones(n))                    # random direction
    delta1 /= torch.norm(delta1)                                                # normalize
    delta2 = torch.normal(torch.zeros(n), std=torch.ones(n))
    delta2 /= torch.norm(delta2)
    
    ### split direction vector into subparts fitting parameters' shapes
    delta1_sub = [ delta1[0:n_list[0]].view(params[0].shape).clone() ]
    delta2_sub = [ delta2[0:n_list[0]].view(params[0].shape).clone() ]
    for i in range(1, N):
        indx_start = np.sum(n_list[0:i])
        indx_end = indx_start + n_list[i]
        delta1_sub.append( delta1[indx_start : indx_end].view(params[i].shape).clone() )
        delta2_sub.append( delta2[indx_start : indx_end].view(params[i].shape).clone() )
        
    
    ### store loss values along random direction
    loss_vals = np.zeros((N_grid,N_grid))
    alpha_vals1 = np.linspace(-alpha0, alpha0, N_grid)
    alpha_vals2 = np.linspace(-alpha0, alpha0, N_grid)
    dummy_model = copy.deepcopy(model)                                          # make a model copy to evaluate loss surface
    dummy_state_dict = dummy_model.state_dict()
    param_keys = list(dummy_state_dict.keys())
    dummy_params = [0]*N
    with torch.no_grad():
        for a1 in range(0,len(alpha_vals1)):
            alpha1 = alpha_vals1[a1]
            for a2 in range(0,len(alpha_vals2)):
                alpha2 = alpha_vals2[a2]
                ### perturb network parameters along random direction
                for i in range(0,N):
                    dummy_params[i] = params[i] + alpha1*delta1_sub[i] + alpha2*delta2_sub[i]
                    dummy_state_dict[param_keys[i]] = dummy_params[i]
          
                dummy_model.load_state_dict(dummy_state_dict)                   # load perturbed parameters into dummy model
                loss_vals[a1][a2] = dummy_model.test(batches_train, criterion)  # evaluate model
            

    print("...took {} minutes".format( (time.time()-start_time)/60 ) )    

    alpha_vals2, alpha_vals1 = np.meshgrid(alpha_vals2, alpha_vals1)  # CAREFUL WITH ORDER!
    
    ### plot results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(alpha_vals2, alpha_vals1, loss_vals)
    ax.set_xlabel("\n" + r"$\alpha_2$")
    ax.set_ylabel("\n" + r"$\alpha_1$")
    ax.set_zlabel("\n loss")
    plt.title(r"$L(\theta^{*} + \alpha_1 \theta_{{r1}} + \alpha_2 \theta_{{r2}})$")     # later: pass title as argument
    plt.show()


    
def plot_loss_start_end_1D(init_models, final_configs, batches_train, labels=[], N_grid=1000):
    """
    Projection of loss surface on direction in parameter space between pairs of parameterized 
    models (typically each pair holds an initialized and the correspondingly trained network).
    Expects list of initialized networks (init_models), and list of parameter-lists of trained networks (final_configs).
    All networks need to have the same shape.
    """
    
    print("Evaluating loss along 1D direction from init to finish...")

    network_idx = len(init_models)                                       # no. of networks to be considered     
    
    plt.subplots()
    for idx in range(0, network_idx):
        dummy_model = copy.deepcopy(init_models[idx])                    # make a model copy to evaluate loss surface
        dummy_state_dict = dummy_model.state_dict()
        param_keys = list(dummy_state_dict.keys())
        
        params_start = list(init_models[idx].parameters())               # list of initial parameters
        N = len(params_start)   
        params_final = final_configs[idx]                                # list of final parameters
        
        loss_vals = []                                                   # store loss values between initial and final configurations
        alpha_vals = np.linspace(0, 1, N_grid)
        dummy_params = [0]*N
        
        with torch.no_grad():
            for alpha in alpha_vals:
                for i in range(0,N):                                     # iterate over parameters of network
                    ### perturb network parameters along direction
                    dummy_params[i] = (1-alpha)*params_start[i] + alpha*params_final[i]
                    dummy_state_dict[param_keys[i]] = dummy_params[i]
                    
                dummy_model.load_state_dict(dummy_state_dict)            # load perturbed parameters into dummy model
                loss_vals.append(dummy_model.test(batches_train)[0])     # evaluate model
                
        plt.plot(alpha_vals, loss_vals, label=labels[idx])
        
    ### plot result
    plt.legend()
    plt.xlabel(r"$\alpha$")
    plt.ylabel("loss")
    plt.title(r"$L((1-\alpha)\theta_i + \alpha  \theta_f)$ for initial"+
              " $\theta_i$ and final $\theta_f$, Spirals, AdLaLa")      # later: pass title as argument
    plt.show()
        
    
