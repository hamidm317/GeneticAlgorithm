import numpy as np
import matplotlib.pyplot as plt

import time

from utils.py import *

number_of_items = 15

weights_vector = np.random.uniform(low = 1, high = number_of_items * 10, size = number_of_items)
utility_vector = np.random.uniform(low = 0, high = 5, size = number_of_items)

capacity = 200

plt.stem(weights_vector)
plt.xlabel("Item Number")
plt.ylabel("Weight")
plt.title("Weights of Items")
plt.show()

plt.stem(utility_vector)
plt.xlabel("Item Number")
plt.ylabel("Utility")
plt.title("Utilities of Items")
plt.show()

fitness_values = []

start = time.time()

for sample in range(1, 2 ** number_of_items):
    
    chromosome = GS_chrome(sample, number_of_items)
    
    fitness_values.append(fitness_function(chromosome, weights_vector, utility_vector, capacity, w_w, u_w, c_w))

end = time.time()    

print("Maximum Fitness is ", np.max(fitness_values), "using", GS_chrome(np.argmax(fitness_values), number_of_items), "in", end - start, "seconds")

plt.plot(fitness_values)
plt.xlabel("All samples")
plt.ylabel("Fitness Values")

plt.title("Fitness values of whole feasible space")
plt.savefig("Fitness values of whole feasible space.jpg")
plt.show()

w_w = 0.1
u_w = 10
c_w = 10

maximum_fitness_values_by_noItems = []
maximum_fitness_chroms_by_noItems = []
time_consumptions_by_noItems = []

for number_of_items in range(5, 25):

    weights_vector = np.random.uniform(low = 1, high = number_of_items * 10, size = number_of_items)
    utility_vector = np.random.uniform(low = 0, high = 5, size = number_of_items)

    capacity = 30 * number_of_items

    fitness_values = []

    start = time.time()

    for sample in range(1, 2 ** number_of_items):

        chromosome = GS_chrome(sample, number_of_items)

        fitness_values.append(fitness_function(chromosome, weights_vector, utility_vector, capacity, w_w, u_w, c_w))

    end = time.time()    

    maximum_fitness_values_by_noItems.append(np.max(fitness_values))
    maximum_fitness_chroms_by_noItems.append(GS_chrome(np.argmax(fitness_values), number_of_items))
    time_consumptions_by_noItems.append(end - start)
    
    print("Maximum Fitness for", number_of_items, "item is ", np.max(fitness_values), "using", GS_chrome(np.argmax(fitness_values), number_of_items), "in", end - start, "seconds")

## This section estimates the logistic regression parameters fitted to the time-item data

y = np.log(np.array(time_consumptions_by_noItems))
n = np.arange(5, 25)

q_est = (len(n) * np.sum(n * y) - np.sum(n) * np.sum(y)) / (len(n) * np.sum(n ** 2) - np.sum(n) ** 2)
a_est = (np.sum(y) * np.sum(n ** 2) - np.sum(n) * np.sum(n * y)) / (len(n) * np.sum(n ** 2) - np.sum(n) ** 2)

print(np.exp(a_est), np.exp(q_est))

trendline = np.exp(a_est) * np.exp(q_est) ** (n)
plt.scatter(n, time_consumptions_by_noItems)
plt.xlabel("Number of Items")
plt.ylabel("Time Elapsed")
plt.title("Elapsed Time in Greedy Search")
plt.plot(n, trendline)
plt.savefig("Elapsed Time in Greedy Search.jpg")
plt.show()

n_s = np.arange(77)
trendline = np.exp(a_est) * np.exp(q_est) ** (n_s - 5)
plt.xlabel("Number of Items")
plt.ylabel("Time Elapsed")
plt.title("Estimated elapsed Time in Greedy Search")
plt.plot(n_s, trendline)
plt.savefig("Estimated elapsed Time in Greedy Search.jpg")
plt.show()

print("For", n_s[-1], "items the elapsed time is", trendline[-1], "second, means about", int(np.round(trendline[-1] / (60 * 60 * 24 * 365))), "years.")
