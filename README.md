# Quantum Kernel Clustering

A Python implementation of quantum-enhanced clustering using kernel methods and quantum circuit computations. This implementation combines quantum computing concepts with traditional clustering techniques to potentially handle complex, non-linear data patterns.

## Overview

The Quantum Kernel Clustering algorithm leverages quantum circuits to compute similarities between data points in a high-dimensional Hilbert space. It uses quantum operations to potentially capture complex patterns that might be difficult to detect with classical clustering methods.

### Key Features

- Quantum circuit-based kernel computation
- Support for different quantum activation functions
- Configurable network architecture
- Detailed convergence tracking and visualization
- Progress monitoring with comprehensive statistics
- Support for both dense and residual quantum network connections

## Installation

```bash
pip install cirq numpy tensorflow networkx matplotlib seaborn tqdm scikit-learn
```

## Quick Start

```python
from quantum_cluster_kernel import QuantumClusterKernel
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Prepare your data
X = your_data  # Shape: (n_samples, n_features)
scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_scaled = scaler.fit_transform(X)

# Initialize quantum kernel
qk = QuantumClusterKernel(
    n_qubits=X.shape[1],  # Number of features
    layer_structure=[2, 2],  # Quantum network architecture
    connection_type='dense',
    activation='quantum_relu',
    verbose=True
)

# Perform clustering
clusters, info = qk.quantum_clustering(
    data=X_scaled,
    n_clusters=2,
    max_iter=100,
    tolerance=1e-4
)

# Visualize results (for 2D data)
qk.visualize_clustering_results(X_scaled, clusters, info)
```

## Technical Details

### Quantum Network Architecture

The implementation uses a quantum circuit-based approach with:
- Configurable number of qubits
- Layered structure with quantum gates
- Entanglement operations between qubits
- Quantum activation functions

### Kernel Computation

The kernel function is computed using:
1. Quantum state preparation through input encoding
2. Application of parametrized quantum circuits
3. State evolution through quantum operations
4. Inner product computation in the quantum state space

### Clustering Process

The algorithm follows these steps:
1. Compute quantum kernel matrix for all data points
2. Initialize random cluster assignments
3. Iteratively update assignments based on quantum kernel distances
4. Track convergence and cluster stability
5. Visualize results and provide detailed statistics

## Components

### QuantumClusterKernel Class

Main class implementing the quantum kernel clustering algorithm:

```python
class QuantumClusterKernel:
    def __init__(self,
                 n_qubits: int,
                 layer_structure: List[int],
                 connection_type: str = 'dense',
                 activation: str = 'quantum_relu',
                 verbose: bool = True)
```

Key Methods:
- `quantum_clustering()`: Performs the clustering
- `compute_kernel_matrix()`: Computes the quantum kernel matrix
- `visualize_clustering_results()`: Creates visualization plots

### Parameters

- `n_qubits`: Number of qubits (should match input dimension)
- `layer_structure`: List defining the quantum network architecture
- `connection_type`: Type of connections between layers ('dense' or 'residual')
- `activation`: Quantum activation function type
- `verbose`: Enable/disable detailed progress output

### Visualization

The implementation provides three visualization plots:
1. Cluster assignments scatter plot
2. Convergence history
3. Cluster sizes evolution

## Output Interpretation

The algorithm provides comprehensive output including:
- Iteration-by-iteration changes in cluster assignments
- Cluster size evolution
- Convergence metrics
- Final cluster distributions
- Visualization of clustering results

Example output:
```
=== Starting Quantum Clustering Process ===
Number of samples: 100
Number of clusters: 2

Iteration 1/100
Changes in cluster assignments: 38 samples
Cluster sizes: [43 57]

...

=== Clustering Complete ===
Final cluster sizes: [39 61]
Total iterations: 4
```

## Advanced Usage

### Custom Quantum Activation Functions

You can implement custom quantum activation functions:

```python
def quantum_activation(self, value: float, qubit: cirq.Qid):
    return [
        cirq.ry(np.pi/4)(qubit),
        cirq.rz(max(0, value))(qubit),
        cirq.ry(-np.pi/4)(qubit)
    ]
```

### Network Architecture Customization

Modify the layer structure for different architectures:

```python
qk = QuantumClusterKernel(
    n_qubits=4,
    layer_structure=[4, 3, 2],  # Deep architecture
    connection_type='residual'   # With skip connections
)
```

## Performance Considerations

- Computation time scales with the number of qubits and layers
- Kernel matrix computation is the most intensive step
- Memory usage depends on dataset size and number of qubits
- Consider using smaller layer structures for larger datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
