# Haryana Quantum Grid Optimization

A hybrid Graph Neural Network (GNN) and quantum-computing framework for
generator commitment and power-grid optimization using a Haryana-localized
IEEE 14-bus topology.

## Project Overview

This project explores how classical graph-based machine learning can provide
a warm-start for a quantum optimization workflow.

The current mid-semester implementation focuses on:

1. Haryana-localized power-grid topology
2. Real generation capacity integration
3. Graph Neural Network representation of the grid
4. Generator activation probability prediction
5. Mapping GNN probabilities to quantum rotation angles
6. Five-qubit quantum state initialization
7. Quantum state-vector and measurement-probability visualization

The QUBO, Ising Hamiltonian and QAOA optimization stages are planned for the
end-semester phase.

## Architecture

```text
Haryana Power Grid
        |
        v
Graph Representation
        |
        v
Graph Neural Network
        |
        v
Generator Activation Probabilities
        |
        v
theta_i = 2 asin(sqrt(p_i))
        |
        v
Ry(theta_i)
        |
        v
5-Qubit Quantum State
        |
        v
Quantum Measurement Probabilities
        |
        v
Future: QUBO / Ising / QAOA