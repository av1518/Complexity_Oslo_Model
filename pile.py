import numpy as np
import matplotlib.pyplot as plt

class pile:
    def __init__(self, L = 8, p = 0.5):
        self.L = L
        self.p = p
        self.heights = np.zeros(L)
        self.thresholds = np.random.choice([1,2], L, p = [float(self.p), 1-float(self.p)])
        self.slopes = np.zeros(L)
        self.totalgrains = 0
        self.s = [] #avalanche size list
        self.check = np.zeros(self.L)
        self.height = 0

    def rng(self): #returns 1 random number between 2 with p=(given in init) 
        return np.random.choice([1,2], p = [self.p, 1-self.p])

    def relaxsite(self, site):
        if site == 0:
            self.slopes[site] -= 2
            self.slopes[site+1] += 1
            self.thresholds[site] = self.rng()
        elif site == self.L-1:
            self.slopes[site] -= 1
            self.slopes[site-1] += 1
            self.thresholds[site] = self.rng()
        else:
            self.slopes[site] -= 2
            self.slopes[site-1] += 1
            self.slopes[site+1] += 1
            self.thresholds[site] = self.rng()
            
    def checkallsites(self):
        for i in range(self.L):
            if self.slopes[i] > self.thresholds[i]:
                self.check[i] = True
            else:
                self.check[i] = False

    def drive(self):
        self.slopes[0] += 1
        self.totalgrains += 1 
        self.checkallsites()
        relaxations_for_grain = 0
        # print('check = ', self.check)
        while sum(self.check) >= 1:
            sites_to_relax = np.nonzero(self.check)[0]
            # print('check = ', self.check)
            # print(sites_to_relax)
            for site in sites_to_relax:
                self.relaxsite(site)
                relaxations_for_grain += 1
            self.checkallsites()
        else:
            self.s.append(relaxations_for_grain)
        
    def get_av_list(self):
        print('Avalanche list for number of grains added = ', self.totalgrains)
        return self.s
    
    def get_total_grains(self):
        return self.totalgrains
    
    def reset_av_list(self):
        self.s = []
    
    def reset(self):
        self.heights = np.zeros(self.L)
        self.thresholds = np.random.choice([1,2], self.L, p = [float(self.p), 1-float(self.p)])
        self.slopes = np.zeros(self.L)
        self.totalgrains = 0
        self.s = [] #avalanche size list
        self.check = np.zeros(self.L)
        self.height = 0

            
    def get_heights(self):
        self.heights[self.L-1] = self.slopes[self.L-1]
        for i in np.arange(1,self.L,1):
            self.heights[-i-1] = self.heights[-i] + self.slopes[-i-1]
        return self.heights
    
    def plotheights(self):
        plt.bar(np.arange(1, self.L+1, 1), self.get_heights(), width = 1)
        plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
        plt.xlabel('Site')
        plt.ylabel('Height')
        
    # def getheight1(self):#not efficient
    #     return self.get_heights()[0]
     
    def get_h(self): #should be more efficient than the above
        return sum(self.slopes)
    
    def crossover_time(self):
        '''
        Measures Crossover time for a given pile. 
        how: when sum of total heights = total grains added it means no
        grain has fallen off yet. When this  breaks, that grain number is
        crossover time.
        '''
        self.reset()
        while sum(self.get_heights()) == self.totalgrains:
            self.drive()
        else:
            return self.totalgrains
        
    def get_average_slope(self):
        return np.mean(self.slopes)
            
    