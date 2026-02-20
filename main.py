import numpy as np
import random
import matplotlib.pyplot as plt

def create_matrix_with_spikes(rows, cols, base_values, spike_value, num_spikes):
    """
    Create matrix with spikes and noise
    
    Args:
        rows, cols: matrix size
        base_values: noise [3, 4, 5]
        spike_value: spikes value (15)
        num_spikes: number of spikes
    """
    matrix = np.random.choice(base_values, size=(rows, cols))
    
    total_cells = rows * cols
    indices = np.random.choice(total_cells, num_spikes, replace=False)
    
    for idx in indices:
        row = idx // cols
        col = idx % cols
        matrix[row, col] = spike_value
    
    return matrix


def visualize_matrix(matrix, title="Matrix with spikes"):
    """Matrix visualization with digital map"""
    plt.figure(figsize=(10, 8))
    
    spike_mask = matrix == 15
    
    im = plt.imshow(matrix, cmap='viridis', interpolation='nearest')

    spike_positions = np.where(spike_mask)
    plt.scatter(spike_positions[1], spike_positions[0], 
                color='red', s=50, marker='x', label='Spikes (15)')
    
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.legend()
    plt.show()

matrix = create_matrix_with_spikes(10, 10, [3, 4, 5], 15, 6)
print("Matrix 10x10 with 6 spikes 15:")
print(matrix)
visualize_matrix(matrix, "Use matrix")