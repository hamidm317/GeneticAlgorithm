import numpy as np
import matplotlib.pyplot as plt

import time

from utils.py import *

w_w = 0.1
u_w = 10
c_w = 10

min_N = 5
P_m = 0.3

maximum_fitness_values_by_noItems_GA = []
maximum_fitness_chroms_by_noItems_GA = []
time_consumptions_by_noItems_GA = []

maximum_fitness_values_by_noItems = []
maximum_fitness_chroms_by_noItems = []
time_consumptions_by_noItems = []

N_s = np.arange(5, 24)

for number_of_items in N_s:
    
    weights_vector = np.random.uniform(low = 1, high = number_of_items * 10, size = number_of_items)
    utility_vector = np.random.uniform(low = 0, high = 5, size = number_of_items)

    capacity = 30 * number_of_items
    
    max_Gen = 10 * number_of_items
    
    N = min([number_of_items * 10, int(2 ** (number_of_items - 2))])

    pop = []

    for _ in range(N):

        pop.append(np.mod(np.floor(np.random.uniform(low = 1, high = number_of_items * 10, size = number_of_items)), 2))

    max_fits = []
    best_chroms = []
    tav = 0.05 * max(pop_fitness_eval(pop))
    
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
    
    time_consumptions_by_noItems_GA.append(end - start)
    
    
    print("GA: For", number_of_items, "items,", max(max_fits), "has been acheived in", end - start, "the sample is", best_chroms[int(np.argmax(max_fits))])
    maximum_fitness_chroms_by_noItems_GA.append(best_chroms[int(np.argmax(max_fits))])
    maximum_fitness_values_by_noItems_GA.append(max(max_fits))
    
    
    
    fitness_values = []

    start = time.time()

    for sample in range(1, 2 ** number_of_items):

        chromosome = GS_chrome(sample, number_of_items)

        fitness_values.append(fitness_function(chromosome, weights_vector, utility_vector, capacity, w_w, u_w, c_w))

    end = time.time()    

    maximum_fitness_values_by_noItems.append(np.max(fitness_values))
    maximum_fitness_chroms_by_noItems.append(GS_chrome(np.argmax(fitness_values), number_of_items))
    time_consumptions_by_noItems.append(end - start)
    
    print("GSA: Maximum Fitness for", number_of_items, "item is ", np.max(fitness_values), "using", GS_chrome(np.argmax(fitness_values), number_of_items), "in", end - start, "seconds")

## Comparing Time Computations

plt.plot(N_s, time_consumptions_by_noItems, 'b', label = 'Greedy Search Time')
plt.plot(N_s, time_consumptions_by_noItems_GA, 'r', label = 'Genetic Algorithm Time')
plt.xlabel("Number of Items")
plt.ylabel("Time Elapsed")
plt.title("Comparing Genetic Algorithm and Greedy Search in time")
plt.legend()
plt.savefig("Comparing Genetic Algorithm and Greedy Search in time.jpg")

## Comparing The Accuracy

plt.plot(N_s, maximum_fitness_values_by_noItems, 'b', label = 'Greedy Search Time')
plt.plot(N_s, maximum_fitness_values_by_noItems_GA, 'r', label = 'Genetic Algorithm Time')
plt.xlabel("Number of Items")
plt.ylabel("Value")
plt.title("Comparing Genetic Algorithm and Greedy Search in accuarcy")
plt.legend()
plt.savefig("Comparing Genetic Algorithm and Greedy Search in accuarcy.jpg")
plt.show()


plt.plot(N_s, np.abs(np.array(maximum_fitness_values_by_noItems) - np.array(maximum_fitness_values_by_noItems_GA)) / np.array(maximum_fitness_values_by_noItems))
plt.xlabel("Number of Items")
plt.ylabel("Erorr")
plt.title("Genetic Algorithm Error")
plt.savefig("Genetic Algorithm Error.jpg")
plt.show()
