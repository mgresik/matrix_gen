import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

def generate_spike_pattern(col_len, spike_count, max_spike_width=4):
    
    """
    Docstring for generate_spike_pattern
    
    :param col_len: lenght of column
    :param spike_count: counter of spikes
    :param max_spike_width: maximal widht of spikes
    """
    
    spikes = np.zeros(col_len, dtype=bool)  # Create a column with 0-value bool(false)
    
    for _ in range(spike_count):
        # Random point to start spike
        start_pos = np.random.randint(0, col_len - max_spike_width + 1)
        # Random spike width (1-4 points)
        width = np.random.randint(1, max_spike_width + 1)
        spikes[start_pos:start_pos + width] = True  # state true in point of column
        
        # Add the probability of a very large spike (8-12 points)
        # if np.random.random() < 0.05:  # 5% probability
        #     width = np.random.randint(8, 13)
        #     start_pos = np.random.randint(0, col_len - width + 1)
        #     spikes[start_pos:start_pos + width] = True
    
    return spikes

def generate_matrix(rows, cols, noise_min=3, noise_max=4, spike_value=15, 
                   spike_density=0.05, max_spike_width=4, output_format='npy'):
    """
    Generates a matrix with noise and spikes    
    
    :param rows: number of rows
    :param cols: number of columns
    :param noise_min, noise_max: range of noise values
    :param spike_value: splash value
    :param spike_density: spike density (proportion of cells with spikes)
    :param max_spike_width: maximum splash width
    :param output_format: Output format ('npy', 'csv', 'txt')
    """
    
    print(f"Creating matrix {rows} x {cols}...")
    start_time = datetime.now()
    
    # Create a base matrix with noise
    matrix = np.random.uniform(noise_min, noise_max, (rows, cols))
    
    # Calculating the number of columns with spikes
    cols_with_spikes = int(cols * 0.3)  # 30% columns have a spikes
    
    # Calculating all number spikes
    total_spike_cells = int(rows * cols * spike_density)
    spikes_per_column = max(1, total_spike_cells // cols_with_spikes)
    
    print(f"  Noise: [{noise_min}, {noise_max}]")
    print(f"  Value of spikes: {spike_value}")
    print(f"  Density of spikes: {spike_density:.1%}")
    print(f"  Max width of spikes: {max_spike_width}")
    
    # Take a random columns for spikes
    spike_columns = np.random.choice(cols, cols_with_spikes, replace=False)
    
    # Generate spikes for getting columns
    for col_idx in spike_columns:
        # Generate pattern for spikes in columns
        spike_pattern = generate_spike_pattern(rows, spikes_per_column, max_spike_width)
        
        # Применяем всплески
        matrix[spike_pattern, col_idx] = spike_value
    
    # Create any single spikes in other columns
    single_spikes_count = int(rows * cols * 0.01)  # 1% for single spikes
    for _ in range(single_spikes_count):
        row = np.random.randint(0, rows)
        col = np.random.randint(0, cols)
        matrix[row, col] = spike_value
    
    elapsed_time = datetime.now() - start_time
    print(f"Generated ended in {elapsed_time.total_seconds():.1f} second")
    
    return matrix

def save_matrix(matrix, filename, output_format='npy'):
    """Save matrix in this format"""
    if output_format == 'npy':
        np.save(filename, matrix)
        print(f"Matrix saved as {filename}.npy")
    elif output_format == 'csv':
        np.savetxt(f"{filename}.csv", matrix, delimiter=',', fmt='%.2f')
        print(f"Matrix saved as {filename}.csv")
    elif output_format == 'txt':
        np.savetxt(f"{filename}.txt", matrix, delimiter='\t', fmt='%.2f')
        print(f"Matrix saved as {filename}.txt")

def parse_arguments():
    """Parsing arg in shell"""
    parser = argparse.ArgumentParser(
        description='Generator for big matrix with noise and spikes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Samples:
        python matrix_generator.py --rows 180000 --cols 1000
        python matrix_generator.py --noise-min 2 --noise-max 6 --spike-value 20
        python matrix_generator.py --spike-density 0.1 --max-spike-width 6
        python matrix_generator.py --format csv --output my_matrix
        python matrix_generator.py --seed 42 --memory-efficient
        """
    )
    
    parser.add_argument('--rows', type=int, default=180000,
                       help='Value rows (default: 180000)')
    parser.add_argument('--cols', type=int, default=1000,
                       help='Value columns (default: 1000)')
    
    # Настройки шума
    parser.add_argument('--noise-min', type=float, default=3,
                       help='Minimal value of noise (default: 3)')
    parser.add_argument('--noise-max', type=float, default=4,
                       help='Maximal value of noise (default: 4)')
    parser.add_argument('--noise-type', choices=['uniform', 'normal', 'triangular'], 
                       default='uniform', help='type of noise distribution')
    parser.add_argument('--noise-std', type=float, default=0.5,
                       help='Standard deviation for a normal distribution')
    
    # Настройки всплесков
    parser.add_argument('--spike-value', type=float, default=15,
                       help='Value spikes (default: 15)')
    parser.add_argument('--spike-density', type=float, default=0.05,
                       help='Density spikes (0-1, default: 0.05)')
    parser.add_argument('--max-spike-width', type=int, default=4,
                       help='Maximal value of wigth spikes (default: 4)')
    
    # Дополнительные настройки
    parser.add_argument('--format', choices=['npy', 'csv', 'txt'], default='npy',
                       help='Type of output file (default: npy)')
    parser.add_argument('--output', type=str, default='matrix',
                       help='Name of output file (without expansion)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Seed reproducibility of results')
    parser.add_argument('--memory-efficient', action='store_true',
                       help='Use memory-saving mode')
    
    return parser.parse_args()

def generate_with_noise_type(rows, cols, args):
    """Generate matrix with getting type of noise"""
    
    if args.noise_type == 'uniform':
        return np.random.uniform(args.noise_min, args.noise_max, (rows, cols))
    elif args.noise_type == 'normal':
        mean = (args.noise_min + args.noise_max) / 2
        return np.random.normal(mean, args.noise_std, (rows, cols))
    elif args.noise_type == 'triangular':
        mode = (args.noise_min + args.noise_max) / 2
        return np.random.triangular(args.noise_min, mode, args.noise_max, (rows, cols))

def memory_efficient_generation(args):
    """Memory-saved generating mode"""
    chunk_size = 10000  # Generate a 10000 rows per part
    chunks = (args.rows + chunk_size - 1) // chunk_size
    
    print(f"Memory-saved generating mode: generate in {chunks} stage")
    
    # Create file and save first part
    first_chunk = generate_with_noise_type(min(chunk_size, args.rows), args.cols, args)
    apply_spikes_to_matrix(first_chunk, 0, args)
    
    if args.format == 'npy':
        np.save(f"{args.output}_temp.npy", first_chunk)
        # Create other parts
        for i in range(1, chunks):
            print(f"Checking chank {i+1}/{chunks}")
            chunk = generate_with_noise_type(min(chunk_size, args.rows - i*chunk_size), 
                                           args.cols, args)
            apply_spikes_to_matrix(chunk, i*chunk_size, args)
            
            with open(f"{args.output}_temp.npy", 'ab') as f:
                np.save(f, chunk)
        
        # Move temporary file
        import os
        os.rename(f"{args.output}_temp.npy", f"{args.output}.npy")
    else:
        # For text format make a strings
        with open(f"{args.output}.{args.format}", 'w') as f:
            for i in range(chunks):
                print(f"Checking chank {i+1}/{chunks}")
                chunk = generate_with_noise_type(min(chunk_size, args.rows - i*chunk_size), 
                                               args.cols, args)
                apply_spikes_to_matrix(chunk, i*chunk_size, args)
                
                if args.format == 'csv':
                    np.savetxt(f, chunk, delimiter=',', fmt='%.2f')
                elif args.format == 'txt':
                    np.savetxt(f, chunk, delimiter='\t', fmt='%.2f')

def apply_spikes_to_matrix(matrix, row_offset, args):
    """Applies spikes to a part of the matrix"""
    rows, cols = matrix.shape
    
    cols_with_spikes = int(cols * 0.3)
    total_spike_cells = int(rows * cols * args.spike_density)
    spikes_per_column = max(1, total_spike_cells // cols_with_spikes)
    
    spike_columns = np.random.choice(cols, min(cols_with_spikes, cols), replace=False)
    
    for col_idx in spike_columns:
        max_possible_width = min(args.max_spike_width, rows)
        spike_pattern = generate_spike_pattern(rows, spikes_per_column, max_possible_width)
        matrix[spike_pattern, col_idx] = args.spike_value
    
    single_spikes_count = int(rows * cols * 0.01)
    for _ in range(single_spikes_count):
        row = np.random.randint(0, rows)
        col = np.random.randint(0, cols)
        matrix[row, col] = args.spike_value

def visualize_matrix(matrix, title="Матрица со всплесками"):
    """Визуализация матрицы с цветовой картой"""
    plt.figure(figsize=(10, 8))
    
    # Создаём маску для всплесков
    spike_mask = matrix == 15
    
    # Отображаем матрицу
    im = plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    
    # Выделяем всплески другим цветом
    spike_positions = np.where(spike_mask)
    plt.scatter(spike_positions[1], spike_positions[0], 
                color='red', s=50, marker='x', label='Всплески (15)')
    
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel('Столбцы')
    plt.ylabel('Строки')
    plt.legend()
    plt.show()

def main():
    args = parse_arguments()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"State seed: {args.seed}")
    
    try:
        if args.memory_efficient and args.rows > 50000:
            memory_efficient_generation(args)
        else:
            # Standart generation
            matrix = generate_with_noise_type(args.rows, args.cols, args)
            apply_spikes_to_matrix(matrix, 0, args)
            save_matrix(matrix, args.output, args.format)
        
        print(f"\nMatrix {args.rows}x{args.cols} is created.")
        
        # Print statistics
        print("\nStatistics of matrix:")
        print(f"  Min value: {args.noise_min}")
        print(f"  Max value: {args.spike_value}")
        print(f"  Type of noise: {args.noise_type}")
        visualize_matrix(matrix)

        
    except MemoryError:
        print("\nFault: need more memory!")
        print("Try to --memory-efficient or make matrix small")
        sys.exit(1)

if __name__ == "__main__":
    main()