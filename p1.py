import pile_reattempt as p
import numpy as np
import matplotlib.pyplot as plt
import time as t
from tqdm import tqdm


plot_params = {'axes.labelsize':28,
          'axes.titlesize':23,
          'font.size':23,
          'figure.figsize':[10,11],
          'xtick.major.size':13,
          'xtick.minor.size':5,
          'ytick.major.size': 13,
          'ytick.minor.size': 5,
          'font.family': 'STIXGeneral',
          'mathtext.fontset': 'stix'}
plt.rcParams.update(plot_params)
# plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["lightskyblue", "g", "y", 'red', 'm', 'grey', 'brown'])

import cProfile
#%%
plot_params = {
    'figure.figsize':(12,21),
    'savefig.bbox':"tight",
    'savefig.facecolor':"w",
    'xtick.major.size':12,
    'xtick.minor.size':6,
    'ytick.major.size':12,
    'ytick.minor.size':6,
    'axes.labelsize':20,
    'xtick.labelsize':20,
    'ytick.labelsize':20,
    'legend.fontsize':20,
}

plt.rcParams.update(plot_params)


#%%
L = 8
pi = p.pile(L, 1/2)
print('zths = ', pi.thresholds)

#%% '''
'''
for i in range(100):
    pi.drive()
    # print('Slopes =', pi.slopes)
    # print('Heights =', pi.get_heights())
    pi.plotheights()
    plt.show()
'''
#%% Testing L=16
start = t.time()
pile16 = p.pile(16, 1/2)
heightlist16 = []
for i in range(100000):
    pile16.drive()
    heightlist16.append(pile16.get_h())
    # print(f'{i}/100000')
end = t.time()
print("The time of execution of above program is :", end-start)
#%% L = 32
start = t.time()
pile32 = p.pile(32, 1/2)
heightlist32 = []
for i in tqdm(range(100000)):
    pile32.drive()
    heightlist32.append(pile32.get_h())
    # print(f'{i}/100000')

end = t.time()
print("The time of execution of above program is :", end-start)
#%%
# pile16.plotheights()
reccurentheights16 = heightlist16[10000:]
mean16 = np.mean(reccurentheights16)
plt.plot(np.arange(1,1e5+1,1), heightlist16, label = r'L = 16, p = $\frac{1}{2}$')
plt.axhline(mean16, linestyle = '-.',label = f'Mean recurrent config. height = {mean16}', color = 'black')
plt.xlabel('Grains added')
plt.ylabel('Site 1 Height')

reccurentheights32 = heightlist32[10000:]
mean32 = np.mean(reccurentheights32)
plt.plot(np.arange(1,1e5+1,1), heightlist32, label = r'L = 32, p = $\frac{1}{2}$')
plt.axhline(mean32, linestyle = '-.',label = f'Mean reccurent conf. height = {mean32}', color = 'purple')

plt.legend()
plt.show()
#%% Running profiler

def testing():
    pile32 = p.pile(32, 1/2)
    heightlist32 = []
    for i in range(100000):
        pile32.drive()
        heightlist32.append(pile32.get_h())
        # print(f'{i}/100000')
    return heightlist32

# test = testing()
cProfile.run('testing()')

#%% Test all zth = 1
'''
Testing all thresholds = 1, meaning p = 1 and expect all slopes = 1 in steady state
'''
L = 32
prob = 1
pilez1 = p.pile(L, prob)
for i in range(1500):
    pilez1.drive()
print('Slopes =', pilez1.slopes) # gets back correct results


'''
Testing that for p = 1 avalanche size -> L for large grain number
'''

av_list = pilez1.get_av_list()
plt.plot(np.arange(1,1500+1,1),av_list, label = f'L={L}, p = {prob}') #works as intended 

L = 16
prob = 1
pilez1 = p.pile(L, prob)
for i in range(1500):
    pilez1.drive()
print('Slopes =', pilez1.slopes) # gets back correct results


'''
Testing that for p = 1 avalanche size -> L for large grain number
'''

av_list = pilez1.get_av_list()
plt.plot(np.arange(1,1500+1,1),av_list, label = f'L={L}, p = {prob}') #works as intended 

L = 16
prob = 0
pilez1 = p.pile(L, prob)
for i in range(1500):
    pilez1.drive()
print('Slopes =', pilez1.slopes) # gets back correct results


'''
Testing that for p = 1 avalanche size -> L for large grain number
'''

av_list = pilez1.get_av_list()
plt.plot(np.arange(1,1500+1,1),av_list, label = f'L={L}, p = {prob}', linestyle = '-') #works as intended 



plt.yticks(np.arange(0, 40,8))
plt.xlabel('Grains added', size = 27)
plt.ylabel(r'Avalanche size $s$', size = 30)
plt.legend()
#%% Test all zth = 2
'''
Testing all thresholds = 2, meaning p = 0 and expect all slopes = 2 in steady state
also expect avalache size to be = L in steady state
'''
pilez2 = p.pile(32, p = 0)
for i in range(10000):
    pilez2.drive()
print('Slopes =', pilez2.slopes) # gets back correct results

#%%
'''
Testing all thresholds = 1, meaning p = 1 and expect all slopes = 1 in steady state
'''
L = [16,32] 
prob = 1
height_lists = []

for L in tqdm(L):
    pile = p.pile(L, prob)
    heightlist = []
    for j in (range(1500)):
        pile.drive()
        heightlist.append(pile.get_h())
    height_lists.append(heightlist)
np.save('h_lists_task1_p=1', height_lists)

#%%
# L = [4,8,16,32,64]
L = [16,32] 
heights = np.load('h_lists_task1_p=1.npy', allow_pickle = True) #all heights 
for i in range(len(L)):
    plt.plot(np.arange(1,1500+1,1),heights[i],label = f'L = {L[i]}')
    
plt.legend(title = r'Pile size with $p=1$')
# plt.yticks = np.linspace(0,1500, 2)
plt.xlabel('Grains added')
plt.ylabel('Height h')
#%%
'''
Testing all thresholds = 2, meaning p = 0 and expect all slopes = 1 in steady state
'''
L = [4,8,16,32,64] 
prob = 0
height_lists = []

for L in tqdm(L):
    pile = p.pile(L, prob)
    heightlist = []
    for j in (range(5000)):
        pile.drive()
        heightlist.append(pile.get_h())
    height_lists.append(heightlist)
np.save('h_lists_task1_p=0', height_lists)
#%%

L = [4,8,16,32,64] 
heights = np.load('h_lists_task1_p=0.npy', allow_pickle = True) #all heights 
for i in range(len(L)):
    plt.plot(np.arange(1,5000+1,1),heights[i],label = f'L = {L[i]}')
    
plt.legend(title = r'Pile size with $p=0$')
plt.xlabel('Grains added')
plt.ylabel('Height h')

#%% Final plot for test
L = [8,32]



prob = 1
height_lists_p1 = []

for i in tqdm(range(len(L))):
    pile = p.pile(L[i], prob)
    heightlist = []
    for j in (range(1500)):
        pile.drive()
        heightlist.append(pile.get_h())
    height_lists_p1.append(heightlist)


height_lists_p0 = []
for i in tqdm(range(len(L))):
    pile = p.pile(L[i], 0)
    heightlist = []
    for j in (range(1500)):
        pile.drive()
        heightlist.append(pile.get_h())
    height_lists_p0.append(heightlist)
#%%

plt.plot(np.arange(1,1500+1,1),height_lists_p0[0],label = f'L = {L[1]}, p = 0', color = 'blue')
plt.plot(np.arange(1,1500+1,1),height_lists_p0[1],label = f'L = {L[1]}, p = 0', color = 'green')
plt.plot(np.arange(1,1500+1,1),height_lists_p1[0],label = f'L = {L[i]}, p = 1', linestyle = '-.', color = 'red')
plt.plot(np.arange(1,1500+1,1),height_lists_p1[1],label = f'L = {L[i]}, p = 1', linestyle = '-.', color = 'grey')   

plt.legend()
plt.yticks(np.arange(0, 70,8))
plt.xlabel(r'Grains added')
plt.ylabel(r'Height $h$')
# plt.xticksminor(size = 10)
# plt.xscale('log')
# plt.yscale('log')

#%%
plt.plot(np.arange(1,1500+1,1),height_lists_p0[0],label = f'L = {L[0]}, p = 0')
plt.plot(np.arange(1,1500+1,1),height_lists_p0[1],label = f'L = {L[1]}, p = 0')
plt.plot(np.arange(1,1500+1,1),height_lists_p1[0],label = f'L = {L[0]}, p = 1', linestyle = '-.')
plt.plot(np.arange(1,1500+1,1),height_lists_p1[1],label = f'L = {L[1]}, p = 1', linestyle = '-.')  

plt.legend()
plt.yticks(np.arange(0, 70,8))
plt.xlabel(r'Grains added')
plt.ylabel(r'Height $h$')
# plt.xticksminor(size = 10)
# plt.xscale('log')
# plt.yscale('log')
#%%
test = []
for i in range(33):
    test.append(i)
print(np.sum(test))