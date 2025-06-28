import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------
# 1. Sample Entity Data
# -------------------------
customers = ["C1", "C2", "C3", "C4", "C5"]
sellers = ["S1", "S2"]
brands = {"B1": "Philips", "B2": "GenericCo"}

products = {
    "P1": {"brand": "B1", "seller": "S1"},
    "P2": {"brand": "B1", "seller": "S1"},
    "P3": {"brand": "B2", "seller": "S2"},
    "P4": {"brand": "B2", "seller": "S2"},
}

reviews = [
    {"review_id": "R1", "customer": "C1", "product": "P1", "rating": 5},
    {"review_id": "R2", "customer": "C1", "product": "P2", "rating": 5},
    {"review_id": "R3", "customer": "C2", "product": "P2", "rating": 4},
    {"review_id": "R4", "customer": "C3", "product": "P3", "rating": 1},
    {"review_id": "R5", "customer": "C4", "product": "P4", "rating": 1},
    {"review_id": "R6", "customer": "C5", "product": "P3", "rating": 5},
    {"review_id": "R7", "customer": "C5", "product": "P4", "rating": 5},
]

# -------------------------
# 2. Build Graph
# -------------------------
G = nx.DiGraph()

# Add nodes with type labels
for c in customers:
    G.add_node(c, label="customer")

for s in sellers:
    G.add_node(s, label="seller")

for b_id, b_name in brands.items():
    G.add_node(b_id, label="brand", name=b_name)

for p_id, meta in products.items():
    G.add_node(p_id, label="product")
    G.add_edge(p_id, meta["brand"], type="belongs_to")
    G.add_edge(p_id, meta["seller"], type="sold_by")

for r in reviews:
    r_id = r["review_id"]
    G.add_node(r_id, label="review", rating=r["rating"])
    G.add_edge(r["customer"], r_id, type="wrote")
    G.add_edge(r_id, r["product"], type="about")

# -------------------------
# 3. Streamlit UI
# -------------------------
st.title("🛒 E-Commerce Abuse Graph: Multi-Entity Review Network")

st.markdown("This network graph includes customers, reviews, products, sellers, and brands.")

# -------------------------
# 4. Visualize Graph
# -------------------------
color_map = {
    "customer": "lightblue",
    "review": "orange",
    "product": "lightgreen",
    "seller": "pink",
    "brand": "violet"
}

node_colors = []
for node in G.nodes():
    n_type = G.nodes[node].get("label", "unknown")
    node_colors.append(color_map.get(n_type, "gray"))

# Layout and plotting
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(12, 8))
nx.draw(
    G, pos,
    with_labels=True,
    node_color=node_colors,
    node_size=1000,
    font_size=8,
    font_weight='bold',
    edge_color="gray",
    ax=ax
)

# Optional: edge labels for relationships
edge_labels = nx.get_edge_attributes(G, "type")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

plt.title("Multi-Entity Review Graph", fontsize=14)
st.pyplot(fig)
