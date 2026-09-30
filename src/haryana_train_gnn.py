import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import pennylane as qml

# --- REAL HARYANA CAPACITIES (MW) ---
HARYANA_PLANTS = {
    "Jhajjar": 1500.0,
    "Hisar": 1200.0,
    "Panipat": 710.0,
    "Yamunanagar": 600.0,
    "Faridabad": 432.0
}
TOTAL_CAPACITY = sum(HARYANA_PLANTS.values())

class GridWarmStartGNN(nn.Module):
    """Step 2: The AI Warm-Start (GNN)"""
    def __init__(self, num_features=3, hidden=16):
        super(GridWarmStartGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.predictor = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.predictor(x)).squeeze()

def get_haryana_graph(current_demand):
    """Builds the graph with real capacities and current demand."""
    edge_list = torch.tensor([
        [0, 1], [0, 4], [1, 2], [1, 3], [1, 4],
        [2, 3], [3, 4], [3, 6], [3, 8], [4, 5],
        [5, 10], [5, 11], [5, 12], [6, 7], [6, 8],
        [8, 9], [8, 13], [9, 10], [11, 12], [12, 13]
    ], dtype=torch.long).t()
    edge_index = torch.cat([edge_list, edge_list.flip(0)], dim=1)

    x = torch.zeros((14, 3), dtype=torch.float32)
    
    # Mapping to IEEE 14-bus nodes: 0:Panipat, 1:Faridabad, 2:Hisar, 5:Yamunanagar, 7:Jhajjar
    gen_map = {0: "Panipat", 1: "Faridabad", 2: "Hisar", 5: "Yamunanagar", 7: "Jhajjar"}
    
    for i in range(14):
        if i in gen_map:
            x[i, 0] = 1.0 # Is Generator
            x[i, 1] = HARYANA_PLANTS[gen_map[i]] / TOTAL_CAPACITY # Normalized Capacity
        else:
            x[i, 0] = 0.0 # Is Load
            x[i, 2] = current_demand / TOTAL_CAPACITY # Normalized Demand
            
    return Data(x=x, edge_index=edge_index), gen_map

def train_gnn():
    """Trains the GNN on fluctuating Haryana power demands."""
    model = GridWarmStartGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("=== TRAINING AI ON HARYANA GRID (500 EPOCHS) ===")
    
    for epoch in range(500):
        # Simulate a random summer day demand in Haryana (2500 MW to 4200 MW)
        demand = np.random.uniform(2500, 4200)
        data, _ = get_haryana_graph(demand)
        
        optimizer.zero_grad()
        probs = model(data.x, data.edge_index)
        
        # Simple loss: AI should output higher probabilities when demand is high
        target_activation = demand / TOTAL_CAPACITY
        loss = F.mse_loss(probs.mean(), torch.tensor(target_activation, dtype=torch.float32))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")
            
    return model

if __name__ == "__main__":
    # 1. Train the model
    trained_gnn = train_gnn()
    
    # 2. Test it on a specific scenario: A massive 4,000 MW heatwave load in Haryana
    test_demand = 4000.0
    print(f"\n=== TESTING SCENARIO: {test_demand} MW HEATWAVE LOAD ===")
    
    data, gen_map = get_haryana_graph(test_demand)
    with torch.no_grad():
        final_probs = trained_gnn(data.x, data.edge_index)
    
    gen_indices = list(gen_map.keys())
    probs = final_probs[gen_indices]
    
    # Step 2.3: Map probabilities to quantum rotation angles
    angles = 2.0 * torch.asin(torch.sqrt(probs))
    
    for i, idx in enumerate(gen_indices):
        name = gen_map[idx]
        capacity = HARYANA_PLANTS[name]
        print(f"{name:12s} ({capacity:4.0f} MW) | AI Prob: {probs[i]:.2f} | Quantum Ry Angle: {angles[i]:.4f} rad")

    # Step 3: Quantum State Initialization[cite: 2]
    print("\n=== STEP 3: QUANTUM STATE INITIALIZATION[cite: 2] ===")
    dev = qml.device("default.qubit", wires=5)
    
    @qml.qnode(dev)
    def initialize_quantum_state(angles):
        # Instead of a blind 50/50 guess, apply Ry gates to bias the initial state[cite: 2]
        for i in range(5):
            qml.RY(angles[i], wires=i)
        return qml.state()
        
    state_vector = initialize_quantum_state(angles)
    print("Quantum Circuit successfully initialized with AI Warm-Start angles!")
    print(f"State Vector Shape: {state_vector.shape} (2^5 complex amplitudes)")