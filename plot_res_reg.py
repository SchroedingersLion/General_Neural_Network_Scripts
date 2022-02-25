import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator
import csv
from cycler import cycler

"""
This script plots training and test results on a regression problem
from a csv file which is given in columns with headers "epoch n1 n2... epoch n1 n1...",
where "epoch" gives the time during training and ni are labels (eg. different step sizes)
that are used for plotting the respective column. 
There can be more training rows than test rows. It assumes train and test loss.
"""

name = "/home/rene/PhD/Research/ALA_PENTAMER_ML/2.5k_data/scalings/both_scaled/scaling_WD_eta_mom.csv"   # file to read
plt_title = r"Ala-Pentamer, SGDm WD0.005, 2.5k data, bs=50, both scaled"

### read file
with open(name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    
    N_cols = int(next(csv_reader)[0])                                                                    # no. of columns per train/test in 1st row
    train_results = np.empty((1, N_cols))
    test_results = np.empty((1, N_cols))
    
    row2 = next(csv_reader)                                                                              # labels are stored in 2nd row
    label_lst = row2[1 : N_cols]
    
    csv_reader = csv.reader(csv_file, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        if len(row) == N_cols:
            train_results = np.vstack( (train_results, np.asarray(row[0:N_cols])) )
        else:
            train_results = np.vstack( (train_results, np.asarray(row[0:N_cols])) )
            test_results = np.vstack( (test_results, np.asarray(row[N_cols::])) )
    
train_results = train_results[1::,:]
test_results = test_results[1::,:] 


# %% Plot setup
# plt.close('all')
plt.rcParams.update({'font.size': 20})
plt.rc('legend', fontsize=10)                               # legend fontsize
plt.rcParams['axes.grid'] = True
plt.rc('lines', linewidth=2)

fig, axs = plt.subplots(1,2)                                # prepare plotting

### fix colormap and linestyle for many plots
NUM_COLORS = train_results.shape[1]-1
cm = plt.cm.nipy_spectral
color_list = [cm(i) for i in np.linspace(0, 1,NUM_COLORS)]
if NUM_COLORS % 3 == 0:
    style_list = ['-','--',':']*(NUM_COLORS//3)
elif NUM_COLORS % 3 == 2:
    style_list = ['-','--',':']*(NUM_COLORS//3) + ['-','--']
else:
    style_list = ['-','--',':']*(NUM_COLORS//3) + ['-']    

axs[0].set_prop_cycle( cycler('color', color_list) + cycler('linestyle', style_list) )
axs[1].set_prop_cycle( cycler('color', color_list) + cycler('linestyle', style_list) )

tick_spacing = 0.2

for col_idx in range(1, train_results.shape[1]):
    axs[0].plot(train_results[:,0], train_results[:,col_idx], label=label_lst[col_idx-1])  # plot train 
    axs[1].plot(test_results[:,0], test_results[:,col_idx], label=label_lst[col_idx-1])    # plot test

fig.suptitle(plt_title)        
axs[0].legend()
axs[0].set_ylabel("Train loss")
axs[0].set_yscale("linear")
axs[0].set_xlabel("Epochs")
axs[0].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

axs[1].legend()
axs[1].set_ylabel("Test L2 Error")
axs[1].set_yscale("linear")
axs[1].set_xlabel("Epochs")
axs[1].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.show()
