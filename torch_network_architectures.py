import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

"""
Shallow neural networks to test novel training routines.
Depending on the problem, layers and member functions need
to be adjusted.
"""

class Net(nn.Module):   
    """Feed forward neural network."""

    def __init__(self, nodenr,std=0.0001):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(nodenr[0], nodenr[1])
        self.lin2 = nn.Linear(nodenr[1], nodenr[2])
        # nn.init.normal_(self.lin1.weight, mean=0.0, std=std)
        # nn.init.normal_(self.lin1.bias, mean=0.0, std=std)
        # nn.init.normal_(self.lin2.weight, mean=0.0, std=std)
        # nn.init.normal_(self.lin2.bias, mean=0.0, std=std)

    def forward(self, x):
        # self.train()
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        output = F.log_softmax(x, dim=1)
        # output = torch.sigmoid(x)
        return output
    
    
    def test(self, data_loader, criterion):                                     ## returns network output to given input.
        # self.eval()
        loss = 0
        accu = 0        
        with torch.no_grad():                                                   ## don't keep track of gradients.
            for (features, targets) in data_loader:                             ## iterate over batches.                      
                
                ## FOR IMAGE DATA:
                # output = self.forward(features.squeeze().view((features.shape[0],-1)))                     
                # loss += criterion(output, targets)          
                # accu += np.count_nonzero( targets==torch.argmax(output, dim=1) )  
                
                output = self.forward(features)  
                loss += criterion(output, torch.unsqueeze(targets, dim=1))      ## the unsqueeze is necessary for F.binary_cross_entropy, 
                                                                                ## change otherwise.   
                accu += np.count_nonzero(torch.unsqueeze(targets, dim=1) == 
                                          (output >= 0.5))

        return (loss, accu / len(targets))


               
class Net_large(nn.Module):
    """
    One layer larger than "Net". 
    """

    def __init__(self, nodenr,std=0.0001):
        super(Net_large, self).__init__()
        self.lin1 = nn.Linear(nodenr[0], nodenr[1])
        self.lin2 = nn.Linear(nodenr[1], nodenr[2])
        self.lin3 = nn.Linear(nodenr[2], nodenr[3])

    def forward(self, x):
        # self.train()
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        x = self.lin3(x)
        # output = F.log_softmax(x, dim=1)
        output = torch.sigmoid(x)
        return output
    
    
    def test(self, data_loader, criterion):                                ## returns network output to given input.
        # self.eval()
        loss = 0
        accu = 0        
        with torch.no_grad():                                              ## don't keep track of gradients.
            for (features, targets) in data_loader:                        ## iterate over batches.
                
                # ## FOR IMAGE DATA:
                # output = self.forward(features.squeeze().view((features.shape[0],-1)))                     
                # loss += criterion(output, targets)          
                # accu += np.count_nonzero( targets==torch.argmax(output, dim=1) )  
                
                output = self.forward(features)  
                loss += criterion(output, torch.unsqueeze(targets, dim=1))      ## the unsqueeze is necessary for F.binary_cross_entropy, 
                                                                                ## change otherwise.   
                accu += np.count_nonzero(torch.unsqueeze(targets, dim=1) == 
                                          (output >= 0.5))
 
        return (loss, accu / len(targets))
    


class ConvNet(nn.Module):   
    """
    For image classification.
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=(1,1))
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=(2,2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1))
        self.max2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.fc1 = nn.Linear(2304, 500)
        self.fc2 = nn.Linear(500,10)

    def forward(self, x):
        # self.train()
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.max2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
      
        output = F.log_softmax(x, dim=1)

        return output


    def test(self, data_loader, criterion):                                 ## returns network output to given input.
        self.eval()
        loss = 0
        accu = 0        
        with torch.no_grad():                                               ## don't keep track of gradients.
            for (features, targets) in data_loader:                         ## iterate over batches.
                output = self.forward(features)                             ## get model output.
                                                                            ## add losses of all samples in batch.
                loss += criterion(output, targets)
                _, predicted = torch.max(output,1)
                accu += torch.sum(predicted == targets) 

        return (loss, accu / len(targets))
    
    

def acti_func(x):               # activation function 1/(1+x²).
    res = 1/(1+x**2)
    return res


class Net_molec(nn.Module):
    """
    For free energy regression on molecular data.
    """

    def __init__(self, nodenr):
        super(Net_molec, self).__init__()
        self.lin1 = nn.Linear(nodenr[0], nodenr[1])
        self.lin2 = nn.Linear(nodenr[1], nodenr[2])
        self.lin3 = nn.Linear(nodenr[2], nodenr[3])
        self.lin4 = nn.Linear(nodenr[3], nodenr[4])

    def forward(self, x):
        # self.train()
        x = self.lin1(x)
        # x = torch.relu(x)
        x = acti_func(x)
        x = self.lin2(x)
        # x = torch.relu(x)
        x = acti_func(x)
        x = self.lin3(x)
        # x = torch.relu(x)
        x = acti_func(x)
        x = self.lin4(x)        

        output = x
        return output
    

    def test(self, data_loader, criterion, scale_vals=(0,1)):                   ## returns network output to given input.
        # self.eval()
        l2_error = 0       
        with torch.no_grad():                                                   ## don't keep track of gradients.
            for (features, targets) in data_loader:                             ## iterate over batches.
                output = self.forward(features)                                 ## get model output.
                
                                                                                ## add losses of all samples in batch.
                l2_error += criterion(output*scale_vals[1]+scale_vals[0], 
                                      targets.unsqueeze(dim=1)*scale_vals[1]+scale_vals[0])  
     
        return torch.sqrt( 1/len(data_loader.dataset) * l2_error )
    
    
    
### learnable activations

def sinc_transform(x:torch.Tensor, n, h):
    """
    Used for adaptive activation, sinc approximation.
    gives vector of matrices, where each matrix belongs to a single batch array. 
    matrices hold the 2n+1 sinc( ) values for each of the N elements in single
    batch array. output array is of shape (bs, 2n+1, N). 
    """
    
    out = ( x[:,:,None] - (torch.arange(-n, n+1)*h ).repeat(x.shape[0],x.shape[1],1)) / h
    return torch.sinc(out).permute(0,2,1)

class ada_sinc(nn.Module): 
    """
    An adaptive sinc-activation function (activation with learnable parameter).
    """
    
    def __init__(self, in_features: int, no_funcs: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ada_sinc, self).__init__()
        self.in_features = in_features
        self.no_funcs = no_funcs
        self.Aj = nn.Parameter(torch.empty(2*no_funcs+1, **factory_kwargs))
        # self.h = nn.Parameter(torch.empty(1, **factory_kwargs))
        self.h = np.pi/2 * np.sqrt(1/self.no_funcs)
        self.reset_parameters()
        

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.Aj, 0, 1)
        # nn.init.normal_(self.h, 0, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x, y = input.shape
        if y != self.in_features:
            sys.exit(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
        sinc_sum = self.Aj @ sinc_transform(input, self.no_funcs, self.h)
        
        return sinc_sum



class ada_Net_molec(nn.Module):   
    """
    For free energy regression on molecular data, using adaptive activation
    function "ada_sinc". 
    """
 
    def __init__(self, nodenr, n=10):
        super(ada_Net_molec, self).__init__()
        self.lin1 = nn.Linear(nodenr[0], nodenr[1])
        self.sinc1 = ada_sinc(nodenr[1],n)
        self.lin2 = nn.Linear(nodenr[1], nodenr[2])
        self.sinc2 = ada_sinc(nodenr[2],n)        
        self.lin3 = nn.Linear(nodenr[2], nodenr[3])
        self.sinc3 = ada_sinc(nodenr[3],n)            
        self.lin4 = nn.Linear(nodenr[3], nodenr[4])

    def forward(self, x):
        # self.train()
        x = self.lin1(x)
        x = self.sinc1(x)
        x = self.lin2(x)
        x = self.sinc2(x)
        x = self.lin3(x)
        x = self.sinc3(x)
        x = self.lin4(x)        

        output = x
        return output
    
    def test(self, data_loader, criterion):                                 ## returns network output to given input.
        # self.eval()
        l2_error = 0       
        with torch.no_grad():                                               ## don't keep track of gradients.
            for (features, targets) in data_loader:                         ## iterate over batches.
                output = self.forward(features)                             ## get model output.
                
                                                                            ## add losses of all samples in batch.
                l2_error += criterion(output, targets.unsqueeze(dim=1))  
 
        return torch.sqrt( 1/len(data_loader.dataset) * l2_error )