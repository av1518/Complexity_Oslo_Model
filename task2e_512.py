import pile_reattempt as p
import numpy as np
import matplotlib.pyplot as plt
import time as t
from tqdm import tqdm
from scipy.optimize import curve_fit

plot_params = {'axes.labelsize':28,
          'axes.titlesize':18,
          'font.size':30,
          'figure.figsize':[11,8],
          'xtick.major.size':13,
          'xtick.minor.size':8,
          'ytick.major.size': 16,
          'ytick.minor.size': 12,
          'font.family': 'STIXGeneral',
          'mathtext.fontset': 'stix'}
plt.rcParams.update(plot_params)

#%%
L = [4,8,16,32,64,128,256,512] 
t0 = [i*i for i in L] #take upper bound of t_c (cross over time)


heights = np.load('h_lists_1e6_grains_512.npy', allow_pickle = True) #all heights 
attractor_heights = [] # the heights in steady state i.e after t=L**2 (which is the upper bound of t_c)

for i,t_0 in zip(range(len(L)), t0): #get attractor heights
    at_heights = heights[i,t_0:]
    attractor_heights.append(at_heights) 

    
#%%
h_bars = [np.sum(attractor_heights[i])/len(attractor_heights[i]) for i in range(len(L))] #average attractor_heights for each L
# h_bars_test = [np.mean(attractor_heights[i]) for i in range(len(L))]
h_std = [np.std(attractor_heights[i]) for i in range((len(L)))] #standard dev

unique_h = [np.sort(list(set(attractor_heights[i]))) for i in range((len(L)))] #get unique heights in ascending order

bin_edges = [] #here we need to add a value of +1 to the unique heights to get the bin edges, to get probs
for i in range(len(L)):
    unique_h_i = np.append(unique_h[i], unique_h[i][-1] + 1)
    bin_edges.append(unique_h_i)
#%%
probs = [] #probability corresponding to each each height in unique_h, could make a pandas frame out of this
for i in range(len(L)):
    prob, edges, patch = plt.hist(attractor_heights[i], density = 1, bins = bin_edges[i])
    # plt.show()
    probs.append(prob)
    

#%% Task 2e
'''
Plotting h_bar vs L would give a straight line through the origin (if there were no corrections to scaling)

Devide y-axis with L, i.e plot h_bar/L vs L should give you a constant a_0 i.e a straight horizontal line, 
but can see that it is not, => there are corrections to scaling
'''
L = np.array(L)
h_bars = np.array(h_bars)
plt.plot(L,h_bars/L, 'o')
plt.errorbar(L, h_bars/L, yerr = h_std/L**2, capsize = 10 ,linestyle = 'none', color = 'black', label = 'Standard Deviation')
plt.xlabel('L')
plt.ylabel(r'$\langle h \rangle$/L')
# plt.title('showings signs of correction to scaling')
#%%
L = np.array(L)
h_bars = np.array(h_bars)
plt.plot(L,h_bars, 'o')
plt.errorbar(L, h_bars, yerr = h_std, capsize = 10 ,linestyle = 'none', color = 'black', label = 'Standard Deviation')
plt.xlabel(r'$L$')
plt.ylabel(r'$\langle h \rangle$')
# plt.title('showings signs of correction to scaling')

#%%
L = np.array(L)
h_bars = np.array(h_bars)
plt.plot(L,h_bars, 'o',label = 'data points')
plt.xlabel('L')
plt.ylabel(r'$\langle h \rangle$')
# plt.title('optimised fit')
plt.errorbar(L, h_bars, yerr = h_std, capsize = 10 ,linestyle = 'none', color = 'black', label = 'Standard Deviation')


def f(L, a0, a1, omega1):
    return a0 * L * (1 - a1 * L ** (-omega1) )

params, cov = curve_fit(f, L, h_bars, sigma=(h_std))

plt.plot(np.arange(1, 515,1), f(np.arange(1,515,1),*params), label = 'Optimised fit', color = 'firebrick')
plt.legend()
plt.show()

print('fitted values of a0 = %a, a1 = %a, omega1 = %a' % (params[0], params[1], params[2]))
print('with errors a0_err = %a, a1_err = %a, omega1_err = %a' % (np.sqrt(cov[0,0]), np.sqrt(cov[1,1]), np.sqrt(cov[2,2])))
#%% better way

L = np.array([4,8,16,32,64,128,256,512]) 
a_0 = 1.73
yvals = 1 - h_bars/(a_0*L)
plt.plot(np.log(L),np.log(yvals),'.', ms = 12, label = r'$a_0 = 1.73$', color = 'firebrick')


L = np.array([4,8,16,32,64,128,256,512]) 
a_0 = 1.75
yvals = 1 - h_bars/(a_0*L)
plt.plot(np.log(L),np.log(yvals),'.', ms = 12, label = r'$a_0 = 1.75$', color = 'green')


L = np.array([4,8,16,32,64,128,256,512]) 
a_0 = 1.74
yvals = 1 - h_bars/(a_0*L)
plt.plot(np.log(L),np.log(yvals),'.', ms = 12, label = r'$a_0 = 1.74$', color = 'black')


plt.legend()
plt.ylabel(r'$log(1 - \frac{ \langle h \rangle}{a_0 L})$')
plt.xlabel(r'$log(L)$')

def linear(x,m,c):
    return m*x + c

(omega_1,c), cov = curve_fit(linear,np.log(L),np.log(yvals))
plt.plot(np.log(L), linear(np.log(L),omega_1,c), label = 'Linear fit', color = 'black')
print('omega_1 =', -omega_1, 'pm', np.sqrt(cov[0,0]))
print('a_0 =', 1.74, 'pm', 0.01)
plt.legend()
#%% Task 2f: std vs system size
plt.plot(L,h_std, 'o', color = 'firebrick', label = 'Data')
plt.xlabel('L')
plt.ylabel(r'$\sigma_h$', size = 40)

def std_fit(L, m, n):
    return m * L**n

std_params, std_cov = curve_fit(std_fit, L, h_std)

plt.plot(np.arange(4,515,1), std_fit(np.arange(4,515,1),  *std_params), label = r'Optimised fit $\sigma_h = 0.58L^{0.24}$', color = 'black')
plt.legend(prop={'size':23})
plt.yscale('log')
plt.xscale('log')
# plt.xticks(np.linspace(1, ))
plt.show()
print('fitted values of m = %a, n = %a' % (std_params[0], std_params[1]))
print('with errors n_err = %a, m_err = %a' % (np.sqrt(std_cov[0,0]), np.sqrt(std_cov[1,1])))

#%% 2g: Gaussian probabilities and data collapse
for i in range(len(L)-1):
    plt.plot( unique_h[i], probs[i], label = f'L = {L[i]}', linewidth = 2)
plt.legend(prop={'size':22})
plt.ylabel(r'$P(h;L)$')
plt.xlabel(r'$h$')
plt.xticks(size = 25)
plt.yticks(np.arange(0,0.5,0.05),size = 25)
# plt.grid()

#%% Data collapse@

h_collapse = []
for i in range(len(L)):
    h_col = (unique_h[i] -  h_bars[i])/h_std[i] 
    h_collapse.append(h_col)




p_collapse = []
for i in range(len(L)):
    p_col = h_std[i] * probs[i]
    p_collapse.append(p_col)

for i in range(len(L)):
    plt.plot( h_collapse[i], p_collapse[i],'.', label = f'L = {L[i]}', ms = 18)


plt.ylabel(r'$\sigma_h$P(h;L)', size = 40)
plt.xlabel(r'$\left( h - \langle h \rangle \right) /\sigma_h$', size = 40)
plt.xticks(size = 30)
plt.yticks(size  = 30)

def gaussian(x,sigma,mu):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((x-mu)**2)/sigma**2)

x = np.linspace(-4,4,100)
y = gaussian(x,1,0)
plt.plot(x, y, label = 'Standard Gaussian', linewidth = 2, color = 'black')
plt.legend(prop={'size':21})

#%%
h_collapse = []
for i in range(len(L)):
    h_col = (unique_h[i] -  h_bars[i])/h_std[i]
    h_collapse.append(h_col)




p_collapse = []
for i in range(len(L)):
    p_col = h_std[i] * probs[i]
    p_collapse.append(p_col)

for i in range(len(L)):
    plt.plot( h_collapse[i], p_collapse[i],'.', label = f'L = {L[i]}', ms = 18)


plt.ylabel(r'$\sigma_h$P(h;L)', size = 40)
plt.xlabel(r'$\left( h - \langle h \rangle \right) /\sigma_h$', size = 40)
plt.xticks(size = 30)
plt.yticks(size  = 30)
def gaussian(x,sigma,mu):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((x-mu)**2)/sigma**2)

x = np.linspace(-4,4,100)
y = gaussian(x,1,0)
plt.plot(x, y, label = 'Standard Gaussian', linewidth = 2, color = 'black')
plt.legend(prop={'size':21})
plt.yscale('log')