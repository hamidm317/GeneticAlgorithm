import numpy as np
import matplotlib.pyplot as plt

import time

def fitness_function(chromosome, weights_vector, utility_vector, capacity, w_w, u_w, c_w):
    
    N = len(chromosome)
    capacity_penalty = 0
    
    total_weight = np.matmul(chromosome, weights_vector)
    
    total_util = np.matmul(chromosome, utility_vector)
    
    if total_weight > capacity:
        
        capacity_penalty = c_w * (total_weight - capacity) ** 2
        
    return -1 * total_weight * w_w + total_util * u_w - capacity_penalty

def GS_chrome(sample, number_of_items):
    
    return_vec = np.zeros(number_of_items)
    
    bin_value = bin(sample)[2:]
    N = len(bin_value)
    
    for i in range(N):
        
        return_vec[number_of_items - i - 1] = bin_value[N - i - 1]
        
    return return_vec

def update_pop_size(N, Gen, max_Gen, min_N):
    
    return N

def CDF_calc(array):
    
    tmp = []
    for i in range(len(array)):
        
        tmp.append(np.sum(array[:i]))
        
    return np.array(tmp)

def pop_fitness_eval(pop):
    
    output_vec = []
    
    for chromosome in pop:
        
        output_vec.append(fitness_function(chromosome, weights_vector, utility_vector, capacity, w_w, u_w, c_w))
        
    return np.array(output_vec)

def select_parents(pop, tav):
    
    N = len(pop)
    
    pop_fitness_values = pop_fitness_eval(pop)
    
    pop_prob_vector = np.exp(pop_fitness_values / tav) / np.sum(np.exp(pop_fitness_values / tav))
    
    # print(pop_fitness_values)
    
    parents_seed = np.random.rand(2)
    
    pop_CDF = CDF_calc(pop_prob_vector)
    
    C = []
    
    for i, s_c in enumerate(parents_seed):
        
        C.append(np.argmin(np.abs(s_c - pop_CDF)))
        
    return C[0], C[1]

def mating_output(P1, P2):
    
    cross_point = np.random.uniform(low = 0, high = len(P1))
    
    Offspring = np.zeros(len(P1))
    
    Offspring[:int(cross_point)] = P1[:int(cross_point)]
    Offspring[int(cross_point):] = P2[int(cross_point):]
    
    return Offspring

def mutation_process(O, pm):
    
    O_prime = np.zeros(len(O))
    
    for i in range(len(O)):
        
        if np.random.rand() < pm:
            
            O_prime[i] = 1 - O[i]
            
        else:
            
            O_prime[i] = O[i]
    
    return O_prime

