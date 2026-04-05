import pickle
import networkx as nx
import matplotlib.pyplot as plt

with open("output/graph.pkl", "rb") as f:
    G = pickle.load(f)

pos = nx.spring_layout(G, k=1.2, iterations=100)

# Color mapping
color_map = {
    "Method": "red",
    "Result": "green",
    "Definition": "blue",
    "Dataset": "orange",
    "Observation": "purple"
}

node_colors = []
for node in G.nodes(data=True):
    role = node[1].get("role", "Unknown")
    node_colors.append(color_map.get(role, "gray"))

nx.draw(
    G, pos,
    node_color=node_colors,
    node_size=60,
    edge_color="gray",
    alpha=0.7
)

plt.title("Knowledge Graph (Colored by Scholarly Role)")
plt.show()