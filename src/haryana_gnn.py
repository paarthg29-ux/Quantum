import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np

class GridWarmStartGNN(nn.Module):
    """
    A Graph Convolutional Network that reads the power grid map 
    and predicts which generators should be turned ON.
    """
    def __init__(self, num_node_features=3, hidden_channels=16):
        super(GridWarmStartGNN, self).__init__()
        # Two layers of Graph Convolutions
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Final layer to output a single probability score per node
        self.predictor = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        # Layer 1: Learn from immediate neighbors
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Layer 2: Learn from the broader network
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Output a probability (0.0 to 1.0)
        out = torch.sigmoid(self.predictor(x))
        return out.squeeze()

def setup_haryana_graph_data():
    """Converts our Haryana IEEE-14 map into PyTorch Geometric format."""
    # The 20 transmission lines (edges) from our map
    edge_list = torch.tensor([
        [0, 1], [0, 4], [1, 2], [1, 3], [1, 4],
        [2, 3], [3, 4], [3, 6], [3, 8], [4, 5],
        [5, 10], [5, 11], [5, 12], [6, 7], [6, 8],
        [8, 9], [8, 13], [9, 10], [11, 12], [12, 13]
    ], dtype=torch.long).t()
    
    # Make edges bidrectional (power flows both ways)
    edge_index = torch.cat([edge_list, edge_list.flip(0)], dim=1)

    # Node Features: [Is_Generator, Max_Capacity, Current_Demand]
    # 14 nodes, 3 features each
    x = torch.zeros((14, 3), dtype=torch.float32)
    
    # Let's set the features for our 5 Haryana Generators
    generator_nodes = {
        0: 100.0,  # Panipat
        1: 60.0,   # Faridabad
        2: 50.0,   # Hisar
        5: 40.0,   # Yamunanagar
        7: 30.0    # Jhajjar
    }
    
    for node_id in range(14):
        if node_id in generator_nodes:
            x[node_id, 0] = 1.0  # Yes, it is a generator
            x[node_id, 1] = generator_nodes[node_id] / 280.0  # Normalized capacity
        else:
            # It's a load city (like Gurugram or Rohtak)
            x[node_id, 0] = 0.0
            x[node_id, 2] = np.random.uniform(0.05, 0.2) # Simulated power demand
            
    return Data(x=x, edge_index=edge_index)

def convert_probs_to_quantum_angles(probabilities, generator_indices):
    """
    Translates AI probabilities into Quantum Rotation Angles (Ry).
    Formula: Angle = 2 * arcsin(sqrt(probability))
    """
    gen_probs = probabilities[generator_indices]
    
    # Prevent pure 0 or 1 to avoid math errors in arcsin
    gen_probs = torch.clamp(gen_probs, 1e-4, 1.0 - 1e-4)
    
    # The quantum translation
    angles = 2.0 * torch.asin(torch.sqrt(gen_probs))
    return gen_probs, angles

if __name__ == "__main__":
    # 1. Load the Map
    graph_data = setup_haryana_graph_data()
    
    # 2. Initialize the AI Brain
    # Setting a manual seed so we get consistent results for the quantum circuit later
    torch.manual_seed(42) 
    gnn_model = GridWarmStartGNN(num_node_features=3, hidden_channels=16)
    
    # 3. Ask the AI for a guess (Forward Pass)
    with torch.no_grad():
        all_node_probs = gnn_model(graph_data.x, graph_data.edge_index)
    
    # 4. Extract predictions for our 5 power plants (Nodes 0, 1, 2, 5, 7)
    gen_indices = [0, 1, 2, 5, 7]
    gen_names = ["Panipat", "Faridabad", "Hisar", "Yamunanagar", "Jhajjar"]
    
    probs, angles = convert_probs_to_quantum_angles(all_node_probs, gen_indices)
    
    print("=== AI WARM-START PREDICTIONS ===")
    print("The GNN looked at the grid and made these initial predictions:")
    for name, p, angle in zip(gen_names, probs, angles):
        print(f"{name:12s} -> AI Probability: {p.item():.2f} | Quantum Starting Angle: {angle.item():.4f} rad")