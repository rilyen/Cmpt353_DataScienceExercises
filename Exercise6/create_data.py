import time
import numpy as np
import pandas as pd
from implementations import all_implementations

# Program generates random arrays, and uses time.time to measure the wall-clock time each function takes to sort them. Loops are allowed...

# Need > 40 data points to assume normality (if the results are somewhat-normally distributed)

# Restriction #1: processor time. Any larger is too long
random_array = np.random.randint(10000, size = 10000)  
data_all = []

for i in range(50):
    data_row = []  # collect each row of data with all_implementations = [qs1, qs2, qs3, qs4, qs5, merge1, partition_sort]
    # Restruction #2: run count. Must run each sorting implementation an equal number of times (use a loop)
    for sort in all_implementations:  
        st = time.time()
        res = sort(random_array)  # the sorting algorithm
        en = time.time()
        data_row.append(en-st)  # time to sort for this sorting algorithm
    data_all.append(data_row)  # append the row to to prepare for df
    
data = pd.DataFrame(data_all, columns = ['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort'])
data.to_csv('data.csv', index=False)