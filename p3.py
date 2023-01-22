import pile_reattempt as p
import numpy as np
import matplotlib.pyplot as plt
import time as t
from tqdm import tqdm
from scipy.optimize import curve_fit
from logbin import logbin
from matplotlib.ticker import MultipleLocator


# plot_params = {
#     # "text.usetex": True,
#     'figure.figsize':(14,12),
#     'savefig.bbox':"tight",
#     'savefig.facecolor':"w",
#     'xtick.major.size':12,
#     'xtick.minor.size':6,
#     'ytick.major.size':12,
#     'ytick.minor.size':6,
#     'axes.labelsize':20,
#     'xtick.labelsize':20,
#     'ytick.labelsize':20,
#     'legend.fontsize':20,
#     }
# plt.rcParams.update(plot_params)

#%%
plot_params = {'axes.labelsize':25,
          'axes.titlesize':18,
          'font.size':20,
          'figure.figsize':[10,7],
          'xtick.major.size':10,
          'xtick.minor.size':4,
          'ytick.major.size': 14,
          'ytick.minor.size': 12,
          'font.family': 'STIXGeneral',
          'mathtext.fontset': 'stix'}
plt.rcParams.update(plot_params)
# plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

#%% RUN THIS FOR 1e6 GRAINS AVALANCHE KUST
# ss = np.load('s_lists_1e5_grains.npy', allow_pickle = True) #avalanche sizes for each L
ss = np.load('s_lists_1e6_grains_512.npy', allow_pickle = True)
# ss = np.load('s_lists_1e6PLUSL2_grains_512.npy', allow_pickle = True)
L = [4,8,16,32,64,128,256,512] 
t0 = [i*i for i in L] #take upper bound of tau_c


attractor_ss = []



for i,t_0 in zip(range(len(L)), t0): #get attractor heights
    at_ss = ss[i,t_0:]
    attractor_ss.append(at_ss) 
    
N = [len(attractor_ss[i]) for i in range(len(L))]
#%% RUN THIS FOR 1e6+L**2 GRAINS AVALANCHE KUST
# ss = np.load('s_lists_1e5_grains.npy', allow_pickle = True) #avalanche sizes for each L
# ss = np.load('s_lists_1e6_grains_512.npy', allow_pickle = True)
ss = np.load('s_lists_1e6PLUSL2_grains_512.npy', allow_pickle = True)
L = [4,8,16,32,64,128,256,512] 
t0 = [i*i for i in L] #take upper bound of tau_c


attractor_ss = []



for i,t_0 in zip(range(len(L)), t0): #get attractor heights
    at_ss = ss[i][t_0:]
    attractor_ss.append(at_ss) 
    
N = [len(attractor_ss[i]) for i in range(len(L))]
#%%average ss

averages = [np.mean(i) for i in attractor_ss]


#%% Plotting everything to get idea
'''
for i in range(len(L)):
    plt.plot(np.arange(t0[i],1e5,1), attractor_ss[i])
    plt.title(f'Avalaches in attractor range for L = {L[i]}')
    plt.xlabel('s')
    plt.ylabel('t (= grains added)')
    plt.show()

'''
#%% Get unique s values
unique_s = []
unique_s_counts = []

for i in range(len(L)):
    unique_s_i, unique_s_count_i = np.unique(attractor_ss[i], return_counts = True)
    unique_s.append(unique_s_i)
    unique_s_counts.append(unique_s_count_i)

probs = []
for i in range(len(L)):
    prob_i = unique_s_counts[i]/len(attractor_ss[i])
    probs.append(prob_i)
#%%
unique_s_1e5 = []
unique_s_counts_1e5 = []

for i in range(len(L)):
    unique_s_i, unique_s_count_i = np.unique(attractor_ss[i][:int(1e5)], return_counts = True)
    unique_s_1e5.append(unique_s_i)
    unique_s_counts_1e5.append(unique_s_count_i)

probs_1e5 = []
for i in range(len(L)):
    prob_i = unique_s_counts_1e5[i]/len(attractor_ss[i][:int(1e5)])
    probs_1e5.append(prob_i)

    
# #%%

# plt.plot(unique_s[6][:int(1e5)], probs[6][:int(1e5)],'.')
# plt.yscale('log')
# plt.xscale('log')


# x,y = logbin(attractor_ss[6], scale = 1.2)
# plt.loglog(x,y)

# plt.xlabel('s')
# plt.ylabel(r'$P(s;L)$')

# plt.show()
#%%
plt.plot(unique_s_1e5[-1], probs_1e5[-1],'.', label = r'Data for $N = 10^5$', color = 'green', ms = 3)

plt.plot(unique_s[-1], probs[-1],'.', label = r'Data for $N= 10^6$', color = 'blue', ms = 3)

x_1e5,y_1e5 = logbin(attractor_ss[-1][:int(1e5)], scale = 1.3)


x,y = logbin(attractor_ss[-1], scale = 1.2)
plt.plot(x,y, label = 'logbin for $N = 10^6$', color = 'firebrick', linewidth = 3)
plt.plot(x_1e5,y_1e5, '--', color = 'black',linewidth = 2.4, label = 'logbin for $N = 10^5$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$s$', size = 30)
plt.ylabel(r'$P(s;L)$', size = 30)



plt.legend(prop ={'size':22})
plt.show()



#%%
for i in range(len(L)):
    scale_val = 1.3
    x,y = logbin(attractor_ss[i], scale = scale_val )
    plt.loglog(x,y, label = f'L = {L[i]}')
    plt.legend()
    # plt.title(f'logbin with scale = {scale_val} ')
    
plt.ylabel(r'$\tilde{P}(s;L)$', size = 30)
plt.xlabel(r'$s$', size = 30)
# ml = MultipleLocator(2)
# plt.axes().yaxis.set_minor_locator(ml)
# plt.yticks.minor(size = 30)
# plt.yscale('log')
# plt.xscale('log')

#%% 3a(ii)
'''
need L>>1 and s>>1, use plot of 512
'''
s_256, P_256 = logbin(attractor_ss[-1], scale = 1.2)
plt.loglog(s_256,P_256, '.', label = 'Data exluded in fitting', ms = 10)
print('number of points =', len(s_256))
'''
can see that data points follow a power law decay if we exclude first 2 and last 5
fit to estimate t_s
'''

def power(s,a,tau_s):
    return a * s ** -tau_s
# plt.loglog(s_256[2:-4], P_256[2:-4])

(a, tau_s), cov = curve_fit(power, s_256[14:-16], P_256[14:-16])
print(f'a = {a}, tau_s = {tau_s}')
plt.loglog(s_256[11:-17], P_256[11:-17], '.', color = 'firebrick', label = 'Points included in fitting', ms = 10)
plt.loglog(s_256, power(s_256, a,tau_s), color = 'black', label = 'Fit')


# plt.title(f'logbin with scale = {scale_val} ')
plt.ylabel(r'$\tilde{P}(s;L=512)$')
plt.xlabel(r'$s$')
plt.yscale('log')
plt.xscale('log')
plt.legend()
#%%
for i in range(len(L)):
    scale_val = 1.3
    x,y = logbin(attractor_ss[i], scale = scale_val )
    plt.plot(x,y, label = f'L = {L[i]}')
    # plt.legend()
    # plt.title(f'logbin with scale = {scale_val} ')
    
    plt.ylabel(r'$\tilde{P}(s;L)$')
    plt.xlabel(r'$s$')
    plt.yscale('log')
    plt.xscale('log')
    
s_vals = np.arange(1,int(1e6),1)
plt.loglog(s_vals, power(s_vals,2,tau_s), '--', color = 'black', label = r'$s^{-\tau_{s}}$')
plt.legend(prop={'size': 18})

#%% rescale y axis

for i in range(len(L)):
    scale_val = 1.7
    s_i,p_i = logbin(attractor_ss[i], scale = scale_val )
    plt.plot(s_i,p_i * s_i**tau_s, label = f'L = {L[i]}, N = {N[i]}')
    # plt.legend()
    # plt.title(f'rescaled y values logbin with scale = {scale_val} ')
    plt.ylabel(r'$s^{\tau_s} \tilde{P}(s;L)$')
    plt.xlabel(r'$s$')
    plt.yscale('log')
    plt.xscale('log')
plt.legend(prop={'size': 18})

#%%rescale both y and x
'''
to get D: used D(2-tau_s) = 1

'''
D = 2.17
# D = 1/(2-tau_s)

print(D)
for i in range(len(L)):
    scale_val = 1.3
    s_i,p_i = logbin(attractor_ss[i], scale = scale_val )
    plt.loglog(s_i/(L[i]**D),p_i * s_i**tau_s, label = f'L = {L[i]}',linewidth = 2)

    # plt.title(f'rescaled y values logbin with scale = {scale_val} ')
    plt.ylabel(r'$s^{\tau_s} \tilde{P}(s;L)$', size = 30)
    plt.xlabel(r'$\frac{s}{L^{2.17}}$', size = 30)
plt.legend(prop={'size': 19})
#%% 3b
ks = np.array([1,2,3,4,5])

all_moments = []
# sbar_k1 = 
# sbar_k2 = 
# sbar_k3 = 
# sbar_k4 = 
for k in ks:
    moments = []
    for i in range(len(L)):  
        # s_i = np.array(attractor_ss[i], dtype='int64')
        # s_i_k = s_i**k
        s_k_bar = np.average(np.array(attractor_ss[i], dtype='float64')**k)
        moments.append(s_k_bar)
    all_moments.append(moments)


#%%

s5 = np.array(attractor_ss[6], dtype='float64')
s5_k = s5 **4
for i,j in enumerate(s5_k):
    if j < 0:
        print(i,j)
#%%
def powerlaw(L, a, exponent):
    s_k = a*L**exponent
    return s_k

def linear(x, m, c):
    return m*x + c



L = np.array([4,8,16,32,64,128,256,512] )
exponents = []
exponent_errors = []
for i in range(len(ks)):
    plt.loglog((L),(all_moments[i]),'o', label=f'k = {ks[i]}')
    
    # (a,exponent), covariance = curve_fit(powerlaw, L[-3:], all_moments[i][-3:])
    # exponents.append(exponent)
    (m,c), cov = curve_fit(linear, np.log(L[-4:]), np.log(all_moments[i][-4:]))
    exponents.append(m)
    exponent_errors.append(np.sqrt(cov[0,0]))
    plt.loglog(L, powerlaw(L, np.exp(c), m), '--', color = 'black')


plt.legend(loc=0)
plt.xlabel('$L$')
plt.ylabel(r'$\langle s^k \rangle$')
plt.show()

'''
#get slopes for each straight line = D(1+k-tau_s)
# we know s**k =prop= L^D(1+k-tau_s), fit power law
# fit only to last  3 points since we need L>>1 so only to L = 128,256,512
def power(s,a,tau_s):
    return a * s ** -tau_s
'''
# def linear(x, c, m):
#     return m*x + c

#%%
plt.plot(ks, exponents,'o', label = 'data', ms = 8)
plt.xlabel(r'$k$')
plt.ylabel(r'$D(1 + k - \tau_s$)')
(slope,intercept), lin_cov = curve_fit(linear, ks, exponents, sigma = exponent_errors)
plt.plot(ks, linear(ks, slope, intercept), color = 'black', label = 'fit')
plt.errorbar(ks, linear(ks,slope,intercept), yerr = exponent_errors, capsize = 10 ,linestyle = 'none', color = 'black', label = 'Standard Deviation')
plt.legend()
plt.yticks(np.arange(1,11,1))
print('D = %a +/- %a' % (slope, np.sqrt(lin_cov[0,0])))

tau_s_new = 1 - intercept/D
print(tau_s_new)
#%% Final data collapse


D_final = slope
D = D_final
tau_s = tau_s_new
# D = 2.10
for i in range(len(L)):
    scale_val = 1.3
    s_i,p_i = logbin(attractor_ss[i], scale = scale_val )
    plt.loglog(s_i/(L[i]**D),p_i * s_i**tau_s, label = f'L = {L[i]}',linewidth = 1.8)

    # plt.title(f'rescaled y values logbin with scale = {scale_val} ')
plt.ylabel(r'$s^{\tau_s} \tilde{P}(s;L)$', size = 30)
plt.xlabel(r'$\frac{s}{L^D}$', size = 30)
plt.legend(prop={'size': 18})