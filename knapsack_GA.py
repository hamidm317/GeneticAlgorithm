import numpy as np
import matplotlib.pyplot as plt

import time

from utils.py import *

# initialize population

N = 100

pop = []

for _ in range(N):
    
    pop.append(np.mod(np.floor(np.random.uniform(low = 1, high = number_of_items * 10, size = number_of_items)), 2))
    
max_Gen = 50
min_N = 5
P_m = 0.1
tav = 1000000

max_fits = []
best_chroms = []

start = time.time()

for Gen in range(max_Gen):
    
    N_next = update_pop_size(N, Gen, max_Gen, min_N)
    pop_next = []
    
    for _ in range(N_next):
    
        C1, C2 = select_parents(pop, tav)
        
        # print(C1, C2)
        
        O = mating_output(pop[C1], pop[C2])
        
        O_prime = mutation_process(O, P_m)
        
        pop_next.append(O_prime)
    
    pop = pop_next
    N = N_next
    
    
    # print(len(pop))
    max_fits.append(max(pop_fitness_eval(pop)))
    best_chroms.append(pop[int(np.argmax(pop_fitness_eval(pop)))])

end = time.time()
 
print(max(max_fits), "has been acheived in", end - start)
print("Best sample is", best_chroms[int(np.argmax(max_fits))])
plt.plot(max_fits)
