import numpy as np
import argparse
import matplotlib.pyplot as plt

def generate_spike_pattern(col_len, spike_count, max_spike_width=4):
    
    spikes = np.zeros(col_len, dtype=bool)
    last_value = np.int64(0)

    for _ in range(spike_count):

        start_pos = np.random.randint(0, col_len - max_spike_width + 1)
        width = np.random.randint(1, max_spike_width + 1)
        if start_pos <= last_value:
            # spikes[start_pos] = False
            continue
        else :
            spikes[start_pos:start_pos + width] = True
            last_value = start_pos + width
    
    return spikes


def print_matrix(matrix, title="Matrix"):
    plt.figure(figsize=(10,8))

    spike_mask = matrix == 15
    im = plt.imshow(matrix, cmap='viridis', interpolation='nearest')

    spike_position = np.where(spike_mask)

    plt.scatter(spike_position[1], spike_position[0], 
                color='red', s=50, marker='o', label='Spikes (15)')
    
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.legend()
    plt.show()

matrix = generate_spike_pattern(20, 15, 4)
matrix2d = np.zeros((1,20), dtype=int)
print(matrix)
matrix2d[0, matrix] = 15
print_matrix(matrix2d)