import pile_reattempt as p
import numpy as np
import matplotlib.pyplot as plt
import time as t
from tqdm import tqdm
from scipy.optimize import curve_fit
from logbin import logbin



# plot_params = {'axes.labelsize':23,
#           'axes.titlesize':18,
#           'font.size':20,
#           'figure.figsize':[14,7],
#           'xtick.major.size':10,
#           'xtick.minor.size':4,
#           'ytick.major.size': 10,
#           'ytick.minor.size': 4,
#           'font.family': 'Times New Roman'}
# plt.rcParams.update(plot_params)
#%%
plot_params = {'axes.labelsize':28,
          'axes.titlesize':18,
          'font.size':22,
          'figure.figsize':[12,9],
          'xtick.major.size':11,
          'xtick.minor.size':8,
          'ytick.major.size': 11,
          'ytick.minor.size': 8,
          'font.family': 'STIXGeneral',
          'mathtext.fontset': 'stix'}
plt.rcParams.update(plot_params)

#%%RUN THIS if you are loading the files from saved file
L = [4,8,16,32,64,128,256,512] 
grains_to_add = [3*i*i for i in L]  # should probably change this? check
M = [100, 50, 20, 10, 10, 5, 1,1] #chosen by intuition , not a particular equation


height_lists = np.load('height_lists_up_512.npy', allow_pickle = True)
heights_512 = np.load('h_lists_1e6_grains_512.npy', allow_pickle = True) [7]

heights = []
for i in range(len(height_lists)):
    heights.append(height_lists[i])
heights.append(heights_512[:3*512**2])


height_lists = heights
#%%
for i in range(len(height_lists)):
    plt.plot((np.arange(1,len(height_lists[i])+1,1))/(L[i]**2), height_lists[i]/L[i], label = f'L = {L[i]},  M = {M[i]}')

plt.xlabel(r'$\frac{t}{L^2}$', size = 35)
plt.ylabel(r' $\frac{\tilde{h}}{L}$      ', rotation = 'horizontal', size = 35)
plt.yscale('log')
plt.xscale('log')
plt.legend(title = 'System Size', prop={'size':20})
#%%
# height_lists = np.load('height_lists_up_256.npy', allow_pickle = True)
for i in range(len(height_lists)):
    plt.plot((np.arange(1,len(height_lists[i])+1,1))/(L[i]**2), height_lists[i]/L[i], label = f'L = {L[i]}')

plt.xlabel(r'$\frac{t}{L^2}$', size = 35)
plt.ylabel(r' $\frac{\tilde{h}}{L}$      ', rotation = 'horizontal', size = 35)
# xaxis.zoom(5)
plt.xlim([0.5,1.5])
plt.ylim([0.9,2])
# plt.yscale('log')
# plt.xscale('log')
plt.legend(title = 'System Size', prop={'size':20})

#%%fit power law
def powerlaw(t, a, exponent):
    s_k = a*t**exponent
    return s_k

def linear(x, m, c):
    return m*x + c

crossover_time_list = np.load('crossover_time_average_10_iterations.npy', allow_pickle = True)
tc_bar = [np.mean(crossover_time_list[i]) for i in range(len(L))]

n = len(L) - 1

t_cross_n = tc_bar[n]
x_cross = t_cross_n/L[n]**2 #used crossovertime average for this size L to pick points up to the point of crossover x (i.e only the 
#transient state)
x_vals = (np.arange(1,len(height_lists[n])+1,1))/(L[n]**2)
y_vals = height_lists[n]/L[n]
# plt.plot(x_vals,y_vals)

x_trans =[]


for i in x_vals:
    if i<x_cross:
        x_trans.append(i)

y_trans = y_vals[:len(x_trans)]
plt.plot(x_trans,y_trans, label = f'Transient for L={L[n]}', linewidth = 3)
plt.xlabel(r'$\frac{t}{L^2}$')
plt.ylabel(r' $\frac{\tilde{h}}{L}$        ', rotation = 'horizontal')
plt.legend()

(a,exponent), cov = curve_fit(powerlaw, x_trans, y_trans)
print(a,exponent)
exp_err = np.sqrt(cov[1,1])
print('Exponent (expected to be 0.5) is = %a +/- %a' % (exponent, exp_err))
plt.plot(x_trans, powerlaw(x_trans, a, exponent), label = 'curve fit', color = 'black', linestyle = '--', linewidth = 3)

plt.legend()

#%%
t_trans = np.array(x_trans) * L[n]**2
h_trans = np.array(y_trans) * L[n]
plt.loglog(t_trans,h_trans, '.', label = 'data for L=512 transient period', linewidth = 3, color ='firebrick')
plt.loglog(t_trans, np.array(powerlaw(t_trans, a, 0.50)), label = r'$\tilde{h} = \alpha\sqrt{t}$ ', linestyle = '--', linewidth = 3, color = 'black')
plt.xlabel('t', size = 30)
plt.ylabel(r'$\tilde{h}$', size = 30)
plt.legend(prop={'size': 24})
