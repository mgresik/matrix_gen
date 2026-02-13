import numpy as np
import random
import matplotlib.pyplot as plt

def create_matrix_with_spikes(rows, cols, base_values, spike_value, num_spikes):
    """
    Создаёт матрицу с базовым шумом и всплесками
    
    Args:
        rows, cols: размеры матрицы
        base_values: список базовых значений [3, 4, 5]
        spike_value: значение всплеска (15)
        num_spikes: количество всплесков
    """
    matrix = np.random.choice(base_values, size=(rows, cols))
    
    total_cells = rows * cols
    indices = np.random.choice(total_cells, num_spikes, replace=False)
    
    for idx in indices:
        row = idx // cols
        col = idx % cols
        matrix[row, col] = spike_value
    
    return matrix


def visualize_matrix(matrix, title="Матрица со всплесками"):
    """Визуализация матрицы с цветовой картой"""
    plt.figure(figsize=(10, 8))
    
    spike_mask = matrix == 15
    
    im = plt.imshow(matrix, cmap='viridis', interpolation='nearest')

    spike_positions = np.where(spike_mask)
    plt.scatter(spike_positions[1], spike_positions[0], 
                color='red', s=50, marker='x', label='Всплески (15)')
    
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel('Столбцы')
    plt.ylabel('Строки')
    plt.legend()
    plt.show()

matrix = create_matrix_with_spikes(10, 10, [3, 4, 5], 15, 6)
print("Матрица 10x10 с 6 всплесками 15:")
print(matrix)
visualize_matrix(matrix, "Use matrix")