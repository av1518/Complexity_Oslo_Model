import pile_reattempt as p
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

# plot_params = {'axes.labelsize':23,
#           'axes.titlesize':18,
#           'font.size':20,
#           'figure.figsize':[14,7],
#           'xtick.major.size':11,
#           'xtick.minor.size':8,
#           'ytick.major.size': 11,
#           'ytick.minor.size': 8,
#           'font.family': 'Times New Roman'}
# plt.rcParams.update(plot_params)

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
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["b", "g", "r", 'turquoise', 'magenta', 'gold','brown', 'grey'])
# plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


#%%Task 2a up to 512 1e6 grains
height_lists = np.load('h_lists_1e6_grains_512.npy', allow_pickle = True) #all heights 

L = [4,8,16,32,64,128,256,512] 
for i in range(len(L)):
    plt.loglog(np.arange(1,1e6+1,1), height_lists[i], label = f'L = {L[i]}', linestyle = '-')


plt.xlabel(r'Time $t$')
plt.ylabel(r'System height $h(t; L)$')
# plt.legend()
plt.legend(prop={'size': 22})

#%% Task 2b: Crossover-time
testpile = p.pile(L=6, p=1/2)
crosstime = testpile.crossover_time()
print(crosstime)
print('slopes = ', testpile.slopes) #gets a crosstime of 37 => expect that at less than 37 grains no grain will fall out
#%%test previous comment statement
testpile = p.pile(L=8, p=1)
for i in range(36):
    testpile.drive()
print('slopes = ', testpile.slopes)
print('heights = ', testpile.get_heights())
testpile.plotheights() # works as expected
#%%Task 2b

L = [4,8,16,32,64,128,256, 512] 
crossover_time_list = []
for length in tqdm(L):
    pile = p.pile(length, 1/2)
    t_c = []
    for i in range(10): # for 10 iterations i.e average of 10 runs
        crosstime = pile.crossover_time()
        t_c.append(crosstime)
    crossover_time_list.append(t_c)
    
tc_bar = [np.mean(crossover_time_list[i]) for i in range(len(L))]
np.save('crossover_times_10_iterations', crossover_time_list)
# np.save('crossover_time_average_10_iterations', tc_bar)
    
#%%
'''
Get means, standard deviation to be used as errorbar, plot
'''
L = np.array([4,8,16,32,64,128,256, 512]) 
crossover_time_list = np.load('crossover_times_10_iterations.npy', allow_pickle = True)
# crossover_time_list = np.load('crossover_time_average_10_iterations.npy', allow_pickle = True)
# L = [4,8,16,32,64,128,256] 
tc_bar = [np.mean(crossover_time_list[i]) for i in range(len(L))]
tc_std = [np.std(crossover_time_list[i]) for i in range(len(L))]

plt.loglog(L, tc_bar, 'o', label = r'$<t_{c}>$ for 10 iterations', linestyle = 'none', color = 'black')
plt.errorbar(L, tc_bar, yerr = tc_std, capsize = 10 ,linestyle = 'none', color = 'black', label = 'Standard Deviation')
plt.xlabel(r'System Size $L$')
plt.ylabel(r'Average cross-over time $\langle t_{c} \rangle$')


def powerlaw(L, a, exponent):
    t_c = a*L**exponent
    return t_c
(a, exponent), cov = curve_fit(powerlaw, L, tc_bar)
# (a, exponent), cov = curve_fit(powerlaw, L, tc_bar , sigma = 1/np.array(, absolute_sigma = True))
print('a= %a, exponent =%a' % (a,exponent))

# plt.loglog(L,powerlaw(L,a,exponent),label = r'$\langle t_c \rangle = %a L^{%a}$ ' % (a, exponent), color = 'firebrick')
plt.loglog(L,powerlaw(L,a,exponent),label = 'Optimised fit', color = 'firebrick')
plt.legend(prop={'size': 24})
print('exponent_error =', np.sqrt(cov[1,1]))
