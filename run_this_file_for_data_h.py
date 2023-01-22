'''
Collect data i.e heights for 1e6 (one million) grains at each size L
'''
import pile_reattempt as p
import numpy as np
import matplotlib.pyplot as plt
import time as t
from tqdm import tqdm


plot_params = {'axes.labelsize':14,
          'axes.titlesize':14,
          'font.size':15,
          'figure.figsize':[12,5.5],
          'font.family': 'Times New Roman'}
plt.rcParams.update(plot_params)
#%%#Task 2a: Pile height over time
'''

Plot height vs time for pile with L=[4,8,16,32,64,128,256] for 1e6 grains 

'''
L = [4,8,16,32,64,128,256] 

height_lists = []

for L in tqdm(L):
    pile = p.pile(L, 1/2)
    heightlist = []
    for j in (range(1000000)):
        pile.drive()
        heightlist.append(pile.get_h())
    height_lists.append(heightlist)
np.save('h_lists_1e6_grains', height_lists)
#%%#Task 2a: Pile height over time
'''

Plot height vs time for pile with L=[4,8,16,32,64,128,256] for 1e6 grains 

'''
L = [4,8,16,32,64,128,256,512] 

height_lists = []

for L in tqdm(L):
    pile = p.pile(L, 1/2)
    heightlist = []
    for j in (range(1000000)):
        pile.drive()
        heightlist.append(pile.get_h())
    height_lists.append(heightlist)
np.save('h_lists_1e6_grains_512', height_lists)

#%% Task 3: get avalanche lists

L = [4,8,16,32,64,128,256]
s_lists = []


for L in tqdm(L):
    pile = p.pile(L, 1/2)
    for j in range(100000):
        pile.drive()
    s_lists.append(pile.get_av_list())
np.save('s_lists_1e5_grains', s_lists)
    
#%%
plt.plot(np.arange(0,1e5,1),s_lists[0])
plt.plot(np.arange(0,1e5,1),s_lists[6])

#%% Task 3: get avalanche lists

L = [4,8,16,32,64,128,256,512]
s_lists = []


for i in tqdm(range(len(L))):
    pile = p.pile(L[i], 1/2)
    for j in range(1000000):
        pile.drive()
    s_lists.append(pile.get_av_list())
np.save('s_lists_1e6_grains_512', s_lists)

#%% Task 3: get avalanche lists +L^2 so that all of them are in total N=10^6

L = [4,8,16,32,64,128,256,512]
s_lists = []


for i in tqdm(range(len(L))):
    pile = p.pile(L[i], 1/2)
    for j in range(1000000 + L[i]**2):
        pile.drive()
    s_lists.append(pile.get_av_list())
np.save('s_lists_1e6PLUSL2_grains_512', s_lists)
    
#%% average slope


L = [64]
average_slope_lists = []


for i in tqdm(range(len(L))):
    pile = p.pile(L[i], 1/2)
    for j in range(L[i]**2 + 5000):
        pile.drive()
        average_slope_lists.append(pile.get_average_slope())
np.save('avg_slope_lists', average_slope_lists)
#%%
attractor_av_slopes = average_slope_lists[L[0]**2+1:]
av_slope = np.mean(attractor_av_slopes)
print(av_slope)



#%% Task 2d
L = [4,8,16,32,64,128,256,512] 
grains_to_add = [3*i*i for i in L]  # should probably change this? check
M = [100, 50, 20, 10, 10, 5, 1] #chosen by intuition , not a particular equation
#%%
average_heights = []


for l,j,m in tqdm(zip(L, grains_to_add, M)):
    heights_j_list = []
    
    for iterations_j in range(m):
        pile = p.pile(l, 1/2)
        heightlist_jth = []
        for k in range(j):
            pile.drive()
            heightlist_jth.append(pile.get_h())
        heights_j_list.append(heightlist_jth)
        
    heights_j_list = np.array(heights_j_list)
    average_height = np.sum(heights_j_list, axis = 0)/m
    average_heights.append(average_height) 
height_lists = average_heights
np.save('height_lists_up_512', height_lists)
#%%

average_heights = []


for l,j,m in tqdm(zip(L, grains_to_add, M)):
    heights_j_list = []
    
    for iterations_j in range(m):
        pile = p.pile(l, 1/2)
        heightlist_jth = []
        for k in range(j):
            pile.drive()
            heightlist_jth.append(pile.get_h())
        heights_j_list.append(heightlist_jth)
        
    heights_j_list = np.array(heights_j_list)
    average_height = np.sum(heights_j_list, axis = 0)/m
    average_heights.append(average_height) 
height_lists = average_heights
np.save('height_lists_up_256', height_lists)
#%% THIS IF FOR 1MIL. TAKES TOO LONG
L = [4,8,16,32,64,128,256,512] 
grains_to_add = [3*i*i for i in L]  # should probably change this? check
M = [5, 5 , 5, 5, 5, 5, 2, 2] #chosen by intuition , not a particular equation
average_heights = []


for l,j,m in tqdm(zip(L, grains_to_add, M)):
    heights_j_list = []
    
    for iterations_j in range(m):
        pile = p.pile(l, 1/2)
        heightlist_jth = []
        for k in range(int(1e6)):
            pile.drive()
            heightlist_jth.append(pile.get_h())
        heights_j_list.append(heightlist_jth)
        
    heights_j_list = np.array(heights_j_list)
    average_height = np.sum(heights_j_list, axis = 0)/m
    average_heights.append(average_height) 
height_lists = average_heights
np.save('height_lists_up_512_1e6', height_lists)

#%%Plot them and scale them
for i in range(len(height_lists)):
    plt.plot((np.arange(1,len(height_lists[i])+1,1))/(L[i]**2), height_lists[i]/L[i], label = f'L = {L[i]},  M = {M[i]}')

plt.xlabel(r'$\frac{t}{L^2}$', size = 35)
plt.ylabel(r' $\frac{\tilde{h}}{L}$      ', rotation = 'horizontal', size = 35)
plt.yscale('log')
plt.xscale('log')
plt.legend(prop={'size':20})

#%% Task 3: get avalanche lists 3 MORE TIMES TO GET A STANDARD DEV

L = [4,8,16,32,64,128,256,512]
s_lists_1 = []
s_lists_2 = []
s_lists_3 = []
# s_lists_4 = []


for i in tqdm(range(len(L))):
    pile = p.pile(L[i], 1/2)
    for j in range(1000000):
        pile.drive()
    s_lists_1.append(pile.get_av_list())
np.save('s_lists_1e6_grains_512_1', s_lists_1)

for i in tqdm(range(len(L))):
    pile = p.pile(L[i], 1/2)
    for j in range(1000000):
        pile.drive()
    s_lists_2.append(pile.get_av_list())
np.save('s_lists_1e6_grains_512_2', s_lists_2)

for i in tqdm(range(len(L))):
    pile = p.pile(L[i], 1/2)
    for j in range(1000000):
        pile.drive()
    s_lists_3.append(pile.get_av_list())
np.save('s_lists_1e6_grains_512_3', s_lists_3)
