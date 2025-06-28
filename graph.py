import networkx as nx
import matplotlib.pyplot as plt

# Create two groups: one normal, one suspicious (review ring)

G = nx.Graph()

# --- Normal behavior: sparse connections ---
# Customers: A, B, C
# Products: P1, P2, P3

# A reviews P1 and P2
# B reviews P2
# C reviews P3

normal_edges = [
    ('A', 'P1'), ('A', 'P2'),
    ('B', 'P2'),
    ('C', 'P3')
]

# --- Review ring: tightly connected reviewers and products ---
# Customers: X, Y, Z
# Products: R1, R2, R3

# X, Y, Z all review R1, R2, and R3

ring_edges = [
    ('X', 'R1'), ('X', 'R2'), ('X', 'R3'),
    ('Y', 'R1'), ('Y', 'R2'), ('Y', 'R3'),
    ('Z', 'R1'), ('Z', 'R2'), ('Z', 'R3')
]

# Add all nodes and edges
G.add_edges_from(normal_edges + ring_edges)

# Set positions
pos = nx.spring_layout(G, seed=42)

# Color nodes: customers = lightblue, products = lightgreen
colors = []
for node in G.nodes():
    if node in ['A', 'B', 'C', 'X', 'Y', 'Z']:
        colors.append('lightblue')  # Customers
    else:
        colors.append('lightgreen')  # Products

# Draw the graph
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1000, font_size=10, font_weight='bold', edge_color='gray')
plt.title("Review Graph: Normal Behavior vs Review Ring")
plt.show()
