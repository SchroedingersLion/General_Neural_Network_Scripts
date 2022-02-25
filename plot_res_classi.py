import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator
import csv
from cycler import cycler

"""
This script plots training and test results on a classification problem from a csv file 
which is given in columns with headers "epoch n1 n2... epoch n1 n1...",
where "epoch" gives the time during training and "ni" are labels (eg. different step sizes)
that are used for plotting the respective column. There can be more training rows than test rows.
First quarter of cols is train loss, next quarter train accuracies, then test losses, then test accuracies 
(if either only losses or accuracies are given, use plot_res_reg.py instead).
"""

name = "/home/rene/PhD/Research/AdLaLa/spiral_AdLaLa_largebatch_noises.csv"     # file to read
plt_title = r"Spirals, AdLaLa, bs=500 (50%) "

# read file
with open(name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    
    N_cols = int(next(csv_reader)[0])                                           # no. of columns per train/test in 1st row
    train_losses = np.empty((1, N_cols))
    train_accus = np.empty((1,N_cols))
    test_accus = np.empty((1,N_cols))
    test_losses = np.empty((1, N_cols))
    
    row2 = next(csv_reader)                                                     # labels stored in 2nd row
    label_lst = row2[1 : N_cols]
    
    csv_reader = csv.reader(csv_file, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        if len(row) == 2*N_cols:
            train_losses = np.vstack( (train_losses, np.asarray(row[0:N_cols])) )
            train_accus = np.vstack( (train_accus, np.asarray(row[N_cols:2*N_cols])) )
        else:
            train_losses = np.vstack( (train_losses, np.asarray(row[0:N_cols])) )
            train_accus = np.vstack( (train_accus, np.asarray(row[N_cols:2*N_cols])) )
            test_losses = np.vstack( (test_losses, np.asarray(row[2*N_cols:3*N_cols])) )
            test_accus = np.vstack( (test_accus, np.asarray(row[3*N_cols:4*N_cols])) )
    
train_losses = train_losses[1::,:]
train_accus = train_accus[1::,:]
test_losses = test_losses[1::,:]
test_accus = test_accus[1::,:] 


# %% Plot setup
# plt.close('all')
plt.rcParams.update({'font.size': 20})
plt.rc('legend', fontsize=10)
plt.rcParams['axes.grid'] = True
plt.rc('lines', linewidth=2)

fig, axs = plt.subplots(2,2)                                                    # prepare plotting

# fix colormap and linestyle for many plots
NUM_COLORS = train_losses.shape[1]-1
cm = plt.cm.nipy_spectral
color_list = [cm(i) for i in np.linspace(0, 1,NUM_COLORS)]
if NUM_COLORS % 3 == 0:
    style_list = ['-','--',':']*(NUM_COLORS//3)
elif NUM_COLORS % 3 == 2:
    style_list = ['-','--',':']*(NUM_COLORS//3) + ['-','--']
else:
    style_list = ['-','--',':']*(NUM_COLORS//3) + ['-']    

axs[0][0].set_prop_cycle( cycler('color', color_list) + cycler('linestyle', style_list) )
axs[0][1].set_prop_cycle( cycler('color', color_list) + cycler('linestyle', style_list) )
axs[1][0].set_prop_cycle( cycler('color', color_list) + cycler('linestyle', style_list) )
axs[1][1].set_prop_cycle( cycler('color', color_list) + cycler('linestyle', style_list) )

tick_spacing = 0.05

for col_idx in range(1, train_losses.shape[1]):
    axs[0][0].plot(train_losses[:,0], train_losses[:,col_idx], label=label_lst[col_idx-1])  # plot train loss
    axs[0][1].plot(test_losses[:,0], test_losses[:,col_idx], label=label_lst[col_idx-1])    # plot test loss
    axs[1][0].plot(train_accus[:,0], train_accus[:,col_idx], label=label_lst[col_idx-1])    # plot train accuracies
    axs[1][1].plot(test_accus[:,0], test_accus[:,col_idx], label=label_lst[col_idx-1])      # plot test accuracies

fig.suptitle(plt_title)        
axs[0][0].legend(loc="upper left")
axs[0][0].set_ylabel("Train loss")
axs[0][0].set_yscale("linear")
# axs[0][0].set_xlabel("Epochs")
# axs[0][0].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

axs[0][1].set_ylabel("Test loss")
axs[0][1].set_yscale("linear")
# axs[0][1].set_xlabel("Epochs")
# axs[0][1].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

axs[1][0].legend(loc="upper left")
axs[1][0].set_ylabel("Train accuracy")
axs[1][0].set_yscale("linear")
axs[1][0].set_xlabel("Epochs")
axs[1][0].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

axs[1][1].set_ylabel("Test accuracy")
axs[1][1].set_yscale("linear")
axs[1][1].set_xlabel("Epochs")
axs[1][1].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.show()
