import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np
import sys


class SGDm_original(Optimizer):
    """ The original SGDm method introduced by Sutskever et. al.. The PyTorch standard routine 
    is equivalent but uses a scaled version of the velocities. It is nonequivalent for decreasing step sizes."""

    def __init__(self, params, lr=required, momentum=0,
                 weight_decay=0, p_scaling = 1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        print("SGDm weight decay does NOT work!!!")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, p_scaling=p_scaling)
        super(SGDm_original, self).__init__(params, defaults)

        # Insert momentum and thermostat as parameter properties into state dict.
        group = self.param_groups[0]  # Treat with AdLa.
        for theta in group["params"]:
            self.state[theta]["momentum_buffer"] = torch.zeros_like(theta)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        mu = group["momentum"]
        h = group["p_scaling"]
        for theta in group['params']:
            if theta.grad is None:
                continue

            p = self.state[theta]["momentum_buffer"]
 
            # perform actual step
            p.mul_(mu/h)
            p.add_(-lr/h * theta.grad)
            theta.add_(h*p)    

        return loss


class OSGLD(Optimizer):
    """The overdamped SGLD optimizer, i.e. SGD + additive noise, see Welling & Teh"""
    
    def __init__(self, params, lr=required, sig=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if sig < 0:
            raise ValueError("Invalid sigma: {}:".format(sig))
        defaults = dict(lr=lr, sig=sig)
        super(OSGLD, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ## IMPLEMENT MODIFICATIONS
        group = self.param_groups[0]
        for p in group['params']:
            if p.grad is None:
                continue           
            d_p = p.grad
            p.add_(d_p, alpha=-group['lr'])  # ordinary SGD
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # noise for p
            p.add_(rand, alpha=group['sig']*np.sqrt(group['lr'])) # add noise

        return loss

    
class OSGLD_s_ineq(Optimizer):
    """OSGLD subject to spherical inequality constraints 
    (Leimkuhler et al., Constrained Based Regularization of Neural Netowrks).
    It takes as input three parameter groups (constrained, unconstrained, and slack)"""
    
    def __init__(self, params, radius=required, lr=required, beta_inv=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if radius is not required and radius < 0.0:
            raise ValueError("Invalid radius: {}".format(radius))
        if beta_inv < 0:
            raise ValueError(r"Invalid $\beta^-1$: {}:".format(beta_inv))
        defaults = dict(lr=lr, beta_inv=beta_inv, radius=radius)
        super(OSGLD_s_ineq, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ## IMPLEMENT MODIFICATIONS
        # unconstrained group
        group = self.param_groups[0]
        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad
            p.add_(d_p, alpha=-group['lr'])  # ordinary SGD
            # rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # noise for p
            # p.add_(rand, alpha=np.sqrt(2*group['lr']*group['beta_inv'])) # add noise

        ## constrained group + slack
        group = self.param_groups[1]
        group_xi = self.param_groups[2]
        for p, xi in zip(group['params'], group_xi['params']):
            if p.grad is None:
                continue
            d_p = p.grad
            p.add_(d_p, alpha=-group['lr'])  # gradient step
            # rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # noise for p
            # p.add_(rand, alpha=np.sqrt(2*group['lr']*group['beta_inv'])) # add noise
            # rand_xi = torch.normal(mean=torch.zeros(xi.shape), std=torch.ones(xi.shape)) # noise for slack
            # xi.add_(rand_xi, alpha=np.sqrt(2*group['lr']*group['beta_inv'])) # add noise            
            scale = group['radius'] / torch.sqrt(torch.sum(p.detach()**2, axis=1) + xi**2)  #scaling factor 
            p.mul_( scale[:,None] )  # scale rows of param
            xi.mul_( scale ) # scale slack
                
        return loss
    
    
class OSGLD_s_ineq_mod(Optimizer):
    """Modified version of OSGLD_s_ineq in which the constraint only changes the constrained
    weights if the euclidean row norm is > R. Otherwise change only slack variable xi.
    It takes as input three parameter groups (constrained unconstrained, and slack)"""
    
    def __init__(self, params, radius=required, lr=required, beta_inv=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if radius is not required and radius < 0.0:
            raise ValueError("Invalid radius: {}".format(radius))
        if beta_inv < 0:
            raise ValueError(r"Invalid $\beta^-1$: {}:".format(beta_inv))
        defaults = dict(lr=lr, beta_inv=beta_inv, radius=radius)
        super(OSGLD_s_ineq_mod, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ## IMPLEMENT MODIFICATIONS
        # unconstrained group
        group = self.param_groups[0]
        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad
            p.add_(d_p, alpha=-group['lr'])  # ordinary SGD
            # rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # noise for p
            # p.add_(rand, alpha=np.sqrt(2*group['lr']*group['beta_inv'])) # add noise

        ## constrained group + slack
        group = self.param_groups[1]
        group_xi = self.param_groups[2]
        for p, xi in zip(group['params'], group_xi['params']):
            if p.grad is None:
                continue
            
            d_p = p.grad
            p.add_(d_p, alpha=-group['lr'])  # gradient step
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # noise for p
            p.add_(rand, alpha=np.sqrt(2*group['lr']*group['beta_inv'])) # add noise
            # rand_xi = torch.normal(mean=torch.zeros(xi.shape), std=torch.ones(xi.shape)) # noise for slack
            # xi.add_(rand_xi, alpha=np.sqrt(2*group['lr']*group['beta_inv'])) # add noise    
            
            sq_sum =  torch.sum(p.detach()**2, axis=1) # get squared norms of rows
            sq_sum_idx = sq_sum >= group['radius']**2 # which row has squared norm > R?

            if torch.any(sq_sum_idx):
                scale = torch.ones(p.size()[0])
                scale[sq_sum_idx] = group['radius'] / torch.sqrt(torch.sum(p.detach()[sq_sum_idx]**2, axis=1) + xi[sq_sum_idx]**2)  #scaling factor 
                p.mul_( scale[:,None] )  # scale rows of param
                xi.mul_( scale ) # scale slack for modified p rows
                xi[~sq_sum_idx] = torch.sqrt(group['radius']**2 - sq_sum[~sq_sum_idx]) # other slacks

            else: # change only slack here 
                xi = torch.sqrt(group['radius']**2 - sq_sum)

                
        return loss
    

class AdLaLa(Optimizer):
    """AdLaLa optimizer as described in eq. (15) in 
    Leimkuhler et al. Partitioned Integrators for Thermodynamic Parameterization of Neural Networks.
    Operates on two parameter groups: One for adaptive Langevin, one for Langevin"""
    
    def __init__(self, params, lr=required, sigma_A=required, eps=required, tau_AdLa=required,
                 tau_La=required, gamma=required):
        
        # check for invalid hyperparameters
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if sigma_A is not required and sigma_A < 0.0:
            raise ValueError("Invalid sigma_A: {}".format(sigma_A))
        if eps is not required and eps < 0.0:
            raise ValueError("Invalid eps: {}".format(eps))
        if tau_AdLa is not required and tau_AdLa < 0.0:
            raise ValueError("Invalid tau for AdLa layers: {}".format(tau_AdLa))
        if tau_La is not required and tau_La < 0.0:
            raise ValueError("Invalid tau for La layers: {}".format(tau_La))
        if gamma is not required and gamma < 0.0:
            raise ValueError("Invalid gamma: {}".format(gamma))           
        
        # add hyperparameters to default for all parameters
        defaults = dict(lr=lr, sigma_A=sigma_A, eps=eps, tau_AdLa=tau_AdLa, tau_La=tau_La, gamma=gamma)
        super(AdLaLa, self).__init__(params, defaults)
        
        # Insert momentum and thermostat as parameter properties into state dict.
        group = self.param_groups[0]  # Treat with AdLa.
        for theta in group["params"]:
            self.state[theta]["momentum_buffer"] = torch.zeros_like(theta)
            self.state[theta]["thermostat_var"] = torch.tensor(0.1)
        
        ## Is there another parameter group? If so, treat it with basic Langevin.
        if len(self.param_groups)>1:
            group = self.param_groups[1]
            for theta in group["params"]:
                self.state[theta]["momentum_buffer"] = torch.zeros_like(theta)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # AdLa group
        group = self.param_groups[0]
        for theta in group['params']:
            if theta.grad is None:
                continue
            
            p = self.state[theta]["momentum_buffer"]
            xi = self.state[theta]["thermostat_var"]
            lr = group["lr"]
            sig = group["sigma_A"]  # strength of additive noise in AdLa
            eps = group["eps"]  # coupling strength to thermostat in Adla
            tau = group["tau_AdLa"]  # temperature for AdLa
                    
            #B step (to h, not h/2)
            p.add_(-lr*theta.grad)
            
            #A step
            theta.add_(lr/2 * p)         

            #C step
            p.mul_(np.exp(-lr/2 * xi))
            
            #D step
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # additive noise 
            p.add_( sig*np.sqrt(lr/2)*rand )
            
            #E step
            shape_p = p.size()
            if len(shape_p)>1:
                xi.add_( lr*eps * ( torch.linalg.norm( p.reshape(shape_p[0]*shape_p[1],1).squeeze() )**2 
                                   - torch.numel(p)*tau) )
            else:
                xi.add_( lr*eps * ( torch.linalg.norm(p)**2
                                   - torch.numel(p)*tau) )
            #D step
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # additive noise 
            p.add_( sig * np.sqrt(lr/2)*rand )    
            
            #C step
            p.mul_( np.exp(-lr/2 * xi) )

            #A step
            theta.add_(lr/2 * p) 
            

        ## Langevin group
        group = self.param_groups[1]
        for theta in group['params']:
            if theta.grad is None:
                continue

            p = self.state[theta]["momentum_buffer"]
            lr = group["lr"]
            tau2 = group["tau_La"]  # temperature for langevin layers
            gamma = group["gamma"]  # friction for langevin layers
            alpha = np.exp(-gamma*lr)
            
            # B step (to full h)
            p.add_(-lr*theta.grad)
            # A step
            theta.add_(lr/2 * p) 
            # O step
            p.mul_(alpha)
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))  # additive noise        
            p.add_( np.sqrt( tau2*(1-alpha**2) ) * rand )
            # A step
            theta.add_(lr/2 * p)            


        return loss
    

class cAdLaLa(Optimizer):
    """AdLaLa with circle constraints to given radius 
    (circle constraints as introduced in Leimkuhler et al. Constrained Based Regularization of Neural Networks).
    Operates on 4 parameter groups: Adaptive Langevin, constrained adaptive Langevin, Langevin, constrained Langevin in that order.
    If one only enters fewer than 4 parameter groups, the remaining parameter groups should be of shape {"params": []}.
    The radii list has to be provided, even when one does not use constraints (use dummy numbers).
    Expects list "radii" that stores an iterable for each constrained group, holding the radii for the parameters in that group"""

    
    def __init__(self, params, radii, lr=required, sigma_A=required, eps=required, tau_AdLa=required,
                  tau_La=required, gamma=required):

        # check for invalid hyperparameters
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if sigma_A is not required and sigma_A < 0.0:
            raise ValueError("Invalid sigma_A: {}".format(sigma_A))
        if eps is not required and eps < 0.0:
            raise ValueError("Invalid eps: {}".format(eps))
        if tau_AdLa is not required and tau_AdLa < 0.0:
            raise ValueError("Invalid tau for AdLa layers: {}".format(tau_AdLa))
        if tau_La is not required and tau_La < 0.0:
            raise ValueError("Invalid tau for La layers: {}".format(tau_La))
        if gamma is not required and gamma < 0.0:
            raise ValueError("Invalid gamma: {}".format(gamma))           
        
        # add hyperparameters to default for all parameters
        defaults = dict(lr=lr, sigma_A=sigma_A, eps=eps, tau_AdLa=tau_AdLa, tau_La=tau_La, gamma=gamma)
        super(cAdLaLa, self).__init__(params, defaults)

        group = self.param_groups[0]  # Treat with AdLa.        
        for theta in group["params"]:     # Insert momentum and thermostat as parameter properties into state dict
            self.state[theta]["momentum_buffer"] = torch.zeros_like(theta)
            self.state[theta]["thermostat_var"] = torch.tensor(0.1)
            
        group = self.param_groups[1]  # Treat with c-AdLa.
        radius_list = radii[0]  # get list of radii for this group
        k = 0  # help idx
        for theta in group["params"]:
            self.state[theta]["momentum_buffer"] = torch.zeros_like(theta)
            self.state[theta]["thermostat_var"] = torch.tensor(0.1)
            self.state[theta]["radius"] = radius_list[k]
            k += 1
            self.state[theta]["slack_var"] = torch.sqrt(self.state[theta]["radius"]**2-theta.detach()**2)  # perform error check!!
            if torch.count_nonzero(torch.isnan(self.state[theta]["slack_var"])) > 0:
                print("Radius too small, slack variable can't be initialized properly.")
                sys.exit()                
            self.state[theta]["slack_mom"] = torch.zeros_like(theta)

        group = self.param_groups[2]  # treat with Langevin
        for theta in group["params"]:
            self.state[theta]["momentum_buffer"] = torch.zeros_like(theta)
            
        group = self.param_groups[3]  # Treat with c-Langevin.
        radius_list = radii[1]  # get list of radii for this group
        k = 0  # help idx
        for theta in group["params"]:
            self.state[theta]["momentum_buffer"] = torch.zeros_like(theta)
            self.state[theta]["radius"] = radius_list[k]
            k += 1
            self.state[theta]["slack_var"] = torch.sqrt(self.state[theta]["radius"]**2-theta.detach()**2)
            if torch.count_nonzero(torch.isnan(self.state[theta]["slack_var"])) > 0:
                print("Radius too small, slack variable can't be initialized properly.")
                sys.exit()   
            self.state[theta]["slack_mom"] = torch.zeros_like(theta)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # AdLa group
        group = self.param_groups[0]
        for theta in group['params']:
            if theta.grad is None:
                continue
            
            p = self.state[theta]["momentum_buffer"]
            xi = self.state[theta]["thermostat_var"]
            lr = group["lr"]
            sig = group["sigma_A"]  # strength of additive noise in AdLa
            eps = group["eps"]  # coupling strength to thermostat in Adla
            tau = group["tau_AdLa"]  # temperature for AdLa

            #B step (to h, not h/2)
            p.add_(-lr*theta.grad)
            
            #A step (to h/2)
            theta.add_(lr/2 * p)         

            #C step (to h/2)
            p.mul_(np.exp(-lr/2 * xi * eps))
            
            #D step (to h/2)
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # additive noise 
            p.add_( sig*np.sqrt(lr/2)*rand )
            
            #E step (to h)
            shape_p = p.size()
            if len(shape_p)>1:
                xi.add_( lr*eps * ( torch.linalg.norm( p.reshape(torch.numel(p),1).squeeze() )**2 
                                    - torch.numel(p)*tau) )
            else:
                xi.add_( lr*eps * ( torch.linalg.norm(p)**2
                                    - torch.numel(p)*tau) )
            #D step (to h/2)
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # additive noise 
            p.add_( sig * np.sqrt(lr/2)*rand )    
            
            #C step (to h/2)
            p.mul_( np.exp(-lr/2 * xi * eps) )

            #A step (to h/2)
            theta.add_(lr/2 * p) 
            
            
        # cAdLa group
        group = self.param_groups[1]
        for theta in group['params']:
            if theta.grad is None:
                continue
            
            p = self.state[theta]["momentum_buffer"]
            xi = self.state[theta]["thermostat_var"]
            r = self.state[theta]["radius"]
            s = self.state[theta]["slack_var"]
            ps = self.state[theta]["slack_mom"]
            lr = group["lr"]
            sig = group["sigma_A"]  # strength of additive noise in AdLa
            eps = group["eps"]  # coupling strength to thermostat in Adla
            tau = group["tau_AdLa"]  # temperature for AdLa


            #B step (to h, not h/2)
            p.add_(-lr * (1-(theta/r)**2) * theta.grad)
            ps.add_(lr/r**2 * theta * s * theta.grad)
            # #B step (to h/2)
            # p.add_(-0.5*lr * (1-(theta/r)**2) * theta.grad)
            # ps.add_(0.5*lr/r**2 * theta * s * theta.grad)

            
            #A step (to h/2)
            w = 1/r**2 * (s*p - theta*ps)
            sin = torch.sin(w*lr/2)
            cos = torch.cos(w*lr/2)
            
            theta_old = theta.detach().clone()
            s_old = s.clone()
            
            theta.mul_(cos)  # now modify parameters.
            theta.add_(sin*s_old)
            s.mul_(cos)
            s.add_(-sin*theta_old)
            
            p.copy_(w * (-sin*theta_old + cos*s_old))  
            ps.copy_(-w * (cos*theta_old + sin*s_old)) 
            

            #C step (to h/2)
            p.mul_(np.exp(-lr/2 * xi * eps))
            ps.mul_(np.exp(-lr/2 * xi * eps))

            
            #D step (to h/2)
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # additive noise
            pbar = p + sig*np.sqrt(lr/2)*rand
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # additive noise
            psbar = ps + sig*np.sqrt(lr/2)*rand
 
            p.copy_( (1-(theta/r)**2)*pbar - 1/r**2 * theta*s*psbar)
            ps.copy_( -1/r**2 * theta*s*pbar + (1-(s/r)**2)*psbar )

            
            #E step (to h)
            shape_p = p.size()
            if len(shape_p)>1:
                xi.add_( lr*eps * ( torch.linalg.norm( p.reshape(torch.numel(p),1).squeeze() )**2 
                                    - torch.numel(p)*tau) )
            else:
                xi.add_( lr*eps * ( torch.linalg.norm(p)**2
                                    - torch.numel(p)*tau) )
                        

            #D step (to h/2)
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # additive noise
            pbar = p + sig*np.sqrt(lr/2)*rand
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape)) # additive noise
            psbar = ps + sig*np.sqrt(lr/2)*rand
 
            p.copy_( (1-(theta/r)**2)*pbar - 1/r**2 * theta*s*psbar)
            ps.copy_( -1/r**2 * theta*s*pbar + (1-(s/r)**2)*psbar )   

            
            #C step (to h/2)
            p.mul_(np.exp(-lr/2 * xi * eps))
            ps.mul_(np.exp(-lr/2 * xi * eps))
            
            
            #A step (to h/2)
            w = 1/r**2 * (s*p - theta*ps)
            sin = torch.sin(w*lr/2)
            cos = torch.cos(w*lr/2)
            
            theta_old = theta.detach().clone()
            s_old = s.clone()
            
            theta.mul_(cos)  # now modify parameters.
            theta.add_(sin*s_old)
            s.mul_(cos)
            s.add_(-sin*theta_old)
            
            p.copy_(w * (-sin*theta_old + cos*s_old))  
            ps.copy_(-w * (cos*theta_old + sin*s_old)) 
            # print("after second A step:\n")
            # print(s*ps + theta*p)
            # print(s**2+theta**2)
            
            # #B step (to h/2)   CARE!! NEED NEW FORCE!
            # p.add_(-0.5*lr * (1-(theta/r)**2) * theta.grad)
            # ps.add_(0.5*lr/r**2 * theta * s * theta.grad)
            
            
        ## Langevin group
        group = self.param_groups[2]
        for theta in group['params']:
            if theta.grad is None:
                continue

            p = self.state[theta]["momentum_buffer"]
            lr = group["lr"]
            tau2 = group["tau_La"]  # temperature for langevin layers
            gamma = group["gamma"]  # friction for langevin layers
            alpha = np.exp(-gamma*lr)
            
            # B step (to full h)
            p.add_(-lr*theta.grad)
            # #B step (to h/2)
            # p.add_(-0.5*lr*theta.grad)
            # A step (to h/2)
            theta.add_(lr/2 * p) 
            # O step (to h)
            p.mul_(alpha)
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))  # additive noise        
            p.add_( np.sqrt( tau2*(1-alpha**2) ) * rand )
            # A step (to h/2)
            theta.add_(lr/2 * p)  
            # #B step (to h/2)   CARE! NEED NEW FORCE!
            # p.add_(-0.5*lr*theta.grad)
            
            
        ## c-Langevin group
        group = self.param_groups[3]
        for theta in group['params']:
            if theta.grad is None:
                continue

            p = self.state[theta]["momentum_buffer"]
            r = self.state[theta]["radius"]
            s = self.state[theta]["slack_var"]
            ps = self.state[theta]["slack_mom"]
            lr = group["lr"]
            tau2 = group["tau_La"]  # temperature for langevin layers
            gamma = group["gamma"]  # friction for langevin layers
            alpha = np.exp(-gamma*lr)
            
            #B step (to h, not h/2)
            p.add_(-lr * (1-(theta/r)**2) * theta.grad)
            ps.add_(lr/r**2 * theta * s * theta.grad)
            # #B step (to h/2)
            # p.add_(-0.5*lr * (1-(theta/r)**2) * theta.grad)
            # ps.add_(0.5*lr/r**2 * theta * s * theta.grad)
            
            #A step (to h/2)
            w = 1/r**2 * (s*p - theta*ps)
            sin = torch.sin(w*lr/2)
            cos = torch.cos(w*lr/2)
            
            theta_old = theta.detach().clone()
            s_old = s.clone()
            
            theta.mul_(cos)  # now modify parameters.
            theta.add_(sin*s_old)
            s.mul_(cos)
            s.add_(-sin*theta_old)
            
            p.copy_(w * (-sin*theta_old + cos*s_old))  
            ps.copy_(-w * (cos*theta_old + sin*s_old)) 
            
            # print("first A step: w,sin,cos,theta:\n")
            # print(sin)
            # print(cos)
            # print(theta)

            
            # O step (to h)
            rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))  # additive noise        
            pbar = p*alpha + np.sqrt( tau2*(1-alpha**2) ) * rand
            rand = torch.normal(mean=torch.zeros(ps.shape), std=torch.ones(ps.shape))  # additive noise
            psbar = ps*alpha + np.sqrt( tau2*(1-alpha**2) ) * rand
            
            p.copy_( (1-(theta/r)**2)*pbar - 1/r**2 * theta*s*psbar)
            ps.copy_( -1/r**2 * theta*s*pbar + (1-(s/r)**2)*psbar )
            
            #A step (to h/2)
            w = 1/r**2 * (s*p - theta*ps)
            sin = torch.sin(w*lr/2)
            cos = torch.cos(w*lr/2)
            
            theta_old = theta.detach().clone()
            s_old = s.clone()
            
            theta.mul_(cos)  # now modify parameters.
            theta.add_(sin*s_old)
            s.mul_(cos)
            s.add_(-sin*theta_old)
            
            p.copy_(w * (-sin*theta_old + cos*s_old))  
            ps.copy_(-w * (cos*theta_old + sin*s_old)) 
            
            # print("second A step: w,sin,cos,theta:\n")
            # print(sin)
            # print(cos)
            # print(theta)
            
            # #B step (to h/2) CARE!!! NEED NEW FORCE!
            # p.add_(-0.5*lr * (1-(theta/r)**2) * theta.grad)
            # ps.add_(0.5*lr/r**2 * theta * s * theta.grad)

        return loss

class OBA_LD(Optimizer):
    """ OBA integrator for Langevin Dynmics"""
    
    def __init__(self, params, lr=required, tau_La=required, gamma=required):
        
        # check for invalid hyperparameters
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if tau_La is not required and tau_La < 0.0:
            raise ValueError("Invalid tau for La layers: {}".format(tau_La))
        if gamma is not required and gamma < 0.0:
            raise ValueError("Invalid gamma: {}".format(gamma))           
        
        # add hyperparameters to default for all parameters
        defaults = dict(lr=lr, tau_La=tau_La, gamma=gamma)
        super(OBA_LD, self).__init__(params, defaults)
        
        # Insert momentum and thermostat as parameter properties into state dict.
        group = self.param_groups[0]  # Treat with AdLa.
        for theta in group["params"]:
            self.state[theta]["momentum_buffer"] = torch.zeros_like(theta)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ## Perform steps
        group = self.param_groups[0]
        lr = group["lr"]
        tau2 = group["tau_La"]  # temperature for langevin layers
        gamma = group["gamma"]  # friction for langevin layers
        alpha = np.exp(-gamma*lr)
        for theta in group['params']:
            if theta.grad is None:
                continue

            p = self.state[theta]["momentum_buffer"]

            # O step
            p.mul_(alpha)
            # rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))  # additive noise        
            # p.add_( np.sqrt( tau2*(1-alpha**2) ) * rand )            
            # B step (to full h)
            p.add_(-lr*theta.grad)
            # A step
            theta.add_(lr * p) 

        return loss