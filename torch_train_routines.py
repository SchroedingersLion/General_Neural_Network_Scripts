import numpy as np
import torch
import time

def train_model_classi(model, optimizer, criterion, epochs, trainloader, scheduler=None, testloader=None, eval_freq=0):
    """
    Training routine for N-class classification problem. Expects log_softmax as network output and criterion nll_loss.
    Design problems: criterion is hardcoded in model.test().  For image data, feature tensor needs to be squeezed
    before feed-in to model.forward().
    """
    
    loss_train = np.zeros(epochs+1)
    accu_train = np.zeros(epochs+1)
    
    loss_test = 0  # dummy in case no testloader was passed
    accu_test = 0
    
    # initial train loss and accu values
    (loss_train[0], accu_train[0]) = model.test(trainloader, criterion)

    # initial test loss and accu values if necessary
    testmode = False
    if testloader is not None:
        assert isinstance(eval_freq ,int) and eval_freq > 0, 'eval_freq must be positive integer!'
        testmode = True
        loss_test = np.zeros(epochs//eval_freq+1)
        accu_test = np.zeros(epochs//eval_freq+1)
        (loss_test[0], accu_test[0]) = model.test(testloader, criterion)
        t = 1 # index count for loss_test and accu_test (see below)

    # training
    print("starting training...")
    start_time = time.time()
    for epoch in np.arange(1,epochs+1):
        print("Epoch ", epoch)
        for (batchidx, (features, targets)) in enumerate(trainloader): 
            
            # output = model(features)
            
            output = model(features.squeeze().    # FOR IMAGES
            view((features.shape[0],-1)))
            
            optimizer.zero_grad()
            loss = criterion(output, targets)
            loss_train[epoch] += loss.detach()

            _, predicted = torch.max(output,1)
            accu_train[epoch] += torch.sum(predicted == targets) / len(targets)
            
            loss.backward()
            optimizer.step()
        if scheduler is not None: scheduler.step()
        
        if testmode==True and epoch % eval_freq==0: # obtain test values
            (loss_test[t], accu_test[t]) = model.test(testloader, criterion)
            t += 1
    
    end_time = time.time()
    print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
          .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/epochs))
    
    return ( loss_train/len(trainloader), accu_train/len(trainloader), 
             loss_test/len(testloader), accu_test/len(testloader) )


def train_model_classi2(model, optimizer, criterion, epochs, trainloader, scheduler=None, testloader=None, eval_freq=0):
    """
    2-class classification problem. Currently expects single sigmoidal output and criterion binary cross entropy loss.
    Design problems: criterion is hardcoded in model.test(). For image data, feature tensor needs to be squeezed
    before feed-in to model.forward().
    """
    
    loss_train = np.zeros(epochs+1)
    accu_train = np.zeros(epochs+1)
    
    loss_test = 0  # dummy in case no testloader was passed
    accu_test = 0
    
    # initial train loss and accu values
    (loss_train[0], accu_train[0]) = model.test(trainloader, criterion)

    # initial test loss and accu values if necessary
    testmode = False
    if testloader is not None:
        assert isinstance(eval_freq ,int) and eval_freq > 0, 'eval_freq must be positive integer!'
        testmode = True
        loss_test = np.zeros(epochs//eval_freq+1)
        accu_test = np.zeros(epochs//eval_freq+1)
        (loss_test[0], accu_test[0]) = model.test(testloader, criterion)
        t = 1 # index count for loss_test and accu_test (see below)

    # training
    print("starting training...")
    start_time = time.time()
    for epoch in np.arange(1,epochs+1):
        # print("Epoch ", epoch)
        for (batchidx, (features, targets)) in enumerate(trainloader):
            output = model(features)
            # output = model(features.squeeze().    # FOR IMAGES
            # view((features.shape[0],-1)))           
            
            optimizer.zero_grad()
            loss = criterion(output, targets.unsqueeze(1))
            loss_train[epoch] += loss.detach()
            accu_train[epoch] += np.count_nonzero(torch.unsqueeze(targets, dim=1) == (output >= 0.5)) /len(targets)
            
            loss.backward()
            optimizer.step()

        if scheduler is not None: scheduler.step()
        
        if testmode==True and epoch % eval_freq==0: # obtain test values
            print("Epoch ", epoch)    
            (loss_test[t], accu_test[t]) = model.test(testloader, criterion)
            t += 1
    
    end_time = time.time()
    print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
          .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/epochs))
    
    return ( loss_train/len(trainloader), accu_train/len(trainloader), 
             loss_test/len(testloader), accu_test/len(testloader) )


def train_model_reg(model, optimizer, criterion, criterion_test, epochs, trainloader, scheduler=None, testloader=None, eval_freq=0, scale_vals=(0,1)):
    """ 
    Training routine for regression problem. 
    Expects float as network output and MSE loss.
    """
    
    (y_mean, y_std) = scale_vals  
    
    loss_train = np.zeros(epochs+1)
    
    l2_error_test = 0  # dummy in case no testloader was passed

    
    # initial train loss 
    for (batchidx, (features, targets)) in enumerate(trainloader): 
        output = model(features)
        loss = torch.sqrt(criterion(output*y_std+y_mean, targets.unsqueeze(dim=1)*y_std+y_mean))
        loss_train[0] += loss.detach()  

    # initial test loss if necessary
    testmode = False
    if testloader is not None:
        assert isinstance(eval_freq ,int) and eval_freq > 0, 'eval_freq must be positive integer!'
        testmode = True
        l2_error_test = np.zeros(epochs//eval_freq+1)
        l2_error_test[0] = model.test(testloader, criterion_test, scale_vals=(y_mean, y_std))
        t = 1 # index count for loss_test and accu_test (see below)

    # training
    print("starting training...")
    start_time = time.time()
    for epoch in np.arange(1,epochs+1):
        for (batchidx, (features, targets)) in enumerate(trainloader): 
            output = model(features)
            optimizer.zero_grad()
            loss = torch.sqrt(criterion(output*y_std+y_mean, targets.unsqueeze(dim=1)*y_std+y_mean))
            loss_train[epoch] += loss.detach()           
            loss.backward()
            optimizer.step()
        if scheduler is not None: 
            scheduler.step()
            # print(optimizer.param_groups[0]["lr"])
        
        if testmode==True and epoch % eval_freq==0: # obtain test values
            print("Epoch ", epoch)
            l2_error_test[t]= model.test(testloader, criterion_test, scale_vals=(y_mean, y_std))
            t += 1
    
    end_time = time.time()
    print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
          .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/epochs))

    return ( loss_train/len(trainloader), l2_error_test)