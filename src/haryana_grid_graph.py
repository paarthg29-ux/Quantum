import networkx as nx
import matplotlib.pyplot as plt

def build_haryana_ieee14_grid():
    """
    Builds the IEEE 14-Bus topology localized to Haryana.
    """
    # Create an empty graph
    G = nx.Graph()

    # Define the 14 substations (hubs)
    hubs = [
        "Panipat (Gen)", "Faridabad (Gen)", "Hisar (Gen)", "Rohtak (Load)", 
        "Gurugram (Load)", "Yamunanagar (Gen)", "Karnal (Load)", "Jhajjar (Gen)", 
        "Sonipat (Load)", "Ambala (Load)", "Kurukshetra (Load)", "Sirsa (Load)", 
        "Bhiwani (Load)", "Rewari (Load)"
    ]
    
    # Add nodes to graph
    for i, name in enumerate(hubs):
        G.add_node(i, name=name)

    # The official 20 transmission line connections of the IEEE 14-Bus system (0-indexed)
    edges = [
        (0, 1), (0, 4), (1, 2), (1, 3), (1, 4),
        (2, 3), (3, 4), (3, 6), (3, 8), (4, 5),
        (5, 10), (5, 11), (5, 12), (6, 7), (6, 8),
        (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)
    ]
    G.add_edges_from(edges)
    
    return G, hubs

def visualize_grid(G, hubs):
    """Draws the power grid network."""
    plt.figure(figsize=(12, 8))
    
    # Mark the 5 generator nodes with a different color
    generator_nodes = [0, 1, 2, 5, 7]
    node_colors = ['#ff7f0e' if i in generator_nodes else '#1f77b4' for i in G.nodes()]
    
    # Get node names for labels
    labels = {i: hubs[i] for i in G.nodes()}
    
    # Draw the graph
    pos = nx.spring_layout(G, seed=42) # Layout algorithm for clean spacing
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")
    
    plt.title("Haryana-Localized IEEE 14-Bus Power Grid", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    grid_graph, hub_names = build_haryana_ieee14_grid()
    print(f"Grid constructed with {grid_graph.number_of_nodes()} hubs and {grid_graph.number_of_edges()} power lines.")
    visualize_grid(grid_graph, hub_names)