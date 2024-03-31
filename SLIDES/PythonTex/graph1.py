import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Generate a random adjacency matrix (assuming undirected graph) with weights
num_nodes = 100
adjacency_matrix = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    num_connections = np.random.randint(1, num_nodes // 20)  # 5 times fewer connections per node
    connected_nodes = np.random.choice(num_nodes, num_connections, replace=False)
    weights = np.random.rand(num_connections) * 10  # Random weights
    adjacency_matrix[i, connected_nodes] = weights
    adjacency_matrix[connected_nodes, i] = weights  # Symmetric matrix

# Convert the adjacency matrix to a graph
G = nx.from_numpy_matrix(adjacency_matrix)

# Plot the graph
pos = nx.circular_layout(G)  # Positions for all nodes in a circle
node_color = [len(list(G.neighbors(node))) for node in G.nodes()]
edge_width = [adjacency_matrix[edge[0], edge[1]] / 10 for edge in G.edges()]

nx.draw(G, pos, node_color=node_color, cmap=plt.cm.Blues,
        node_size=100, edge_color='gray', width=edge_width)

max_connections = max(node_color)

# Add a color bar for the number of connections
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues,norm=plt.Normalize(vmin = 0, vmax=max_connections)), orientation='vertical', shrink=0.7)
cbar.set_label('Number of Connections')

# Save the plot as an image
plt.title('Social Network Graph (Node Color = Number of Connections)')
plt.savefig('social_network_graph.png')
plt.show()
