
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from urllib.parse import urlencode
from collections import defaultdict

st.set_page_config(layout="wide")
st.title("🧠 E-commerce Knowledge Graph Explorer")

# Load Excel data
df = pd.read_excel("Ecommerce_Graph_Data.xlsx", sheet_name=None)
brands = df['Brands']
product_sets = df['ProductSets']
child_products = df['ChildProducts']
sellers = df['Sellers']
offers = df['Offers']
customers = df['Customers']
orders = df['Orders']
order_items = df['OrderItems']
reviews = df['Reviews']

# Build graph
G = nx.DiGraph()

def build_graph():
    for _, row in brands.iterrows():
        G.add_node(row["BrandID"], type='Brand')
    for _, row in product_sets.iterrows():
        G.add_node(row["ProductSetID"], type='ProductSet')
    for _, row in child_products.iterrows():
        G.add_node(row["ChildProductID"], type='ChildProduct')
    for _, row in sellers.iterrows():
        G.add_node(row["SellerID"], type='Seller')
    for _, row in offers.iterrows():
        G.add_node(row["OfferID"], type='Offer')
    for _, row in customers.iterrows():
        G.add_node(row["CustomerID"], type='Customer')
    for _, row in orders.iterrows():
        G.add_node(row["OrderID"], type='Order')
    for _, row in order_items.iterrows():
        G.add_node(row["OrderItemID"], type='OrderItem')
    for _, row in reviews.iterrows():
        G.add_node(row["ReviewID"], type='Review')

    for _, row in product_sets.iterrows():
        G.add_edge(row["BrandID"], row["ProductSetID"], relation="owns")
    for _, row in child_products.iterrows():
        G.add_edge(row["ProductSetID"], row["ChildProductID"], relation="has_variant")
    for _, row in offers.iterrows():
        G.add_edge(row["SellerID"], row["OfferID"], relation="makes_offer")
        G.add_edge(row["OfferID"], row["ChildProductID"], relation="offers_product")
    for _, row in orders.iterrows():
        G.add_edge(row["CustomerID"], row["OrderID"], relation="places_order")
    for _, row in order_items.iterrows():
        G.add_edge(row["OrderID"], row["OrderItemID"], relation="contains_item")
        G.add_edge(row["OrderItemID"], row["OfferID"], relation="item_offer")
    for _, row in reviews.iterrows():
        G.add_edge(row["CustomerID"], row["ReviewID"], relation="writes_review")
        G.add_edge(row["ReviewID"], row["ChildProductID"], relation="reviews_product")

build_graph()

# Parse query parameters
params = st.experimental_get_query_params()
selected_id = params.get("entity_id", [None])[0]
selected_type = params.get("type", [None])[0]
depth = int(params.get("depth", [2])[0])

# Entity selection fallback
entity_types = {
    "Review": reviews["ReviewID"].tolist(),
    "Customer": customers["CustomerID"].tolist(),
    "Seller": sellers["SellerID"].tolist(),
    "ProductSet": product_sets["ProductSetID"].tolist(),
    "ChildProduct": child_products["ChildProductID"].tolist(),
    "Offer": offers["OfferID"].tolist(),
    "Order": orders["OrderID"].tolist()
}

if not selected_id or not selected_type:
    selected_type = st.selectbox("Select Entity Type", list(entity_types.keys()))
    selected_id = st.selectbox(f"Select {selected_type} ID", entity_types[selected_type])
    depth = st.selectbox("Connection Depth", [1, 2, 3, 4, 5], index=1)

# Metadata tracker
def fan_out_graph_with_metadata(center_id, max_depth=2):
    visited = {center_id: {"depth": 0, "reason": "Origin"}}
    current_layer = [center_id]

    for d in range(1, max_depth + 1):
        next_layer = []
        for node in current_layer:
            neighbors = list(G.successors(node)) + list(G.predecessors(node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    if G.has_edge(node, neighbor):
                        rel = G.edges[node, neighbor].get("relation", "")
                        reason = f"{node} → {neighbor} via '{rel}'"
                    elif G.has_edge(neighbor, node):
                        rel = G.edges[neighbor, node].get("relation", "")
                        reason = f"{neighbor} → {node} via '{rel}'"
                    else:
                        reason = "Connected"
                    visited[neighbor] = {"depth": d, "reason": reason}
                    next_layer.append(neighbor)
        current_layer = next_layer
    return visited

# Color legend
color_map = {
    'Brand': 'skyblue',
    'ProductSet': 'lightgreen',
    'ChildProduct': 'pink',
    'Seller': 'orange',
    'Offer': 'lightgrey',
    'Customer': 'yellow',
    'Order': 'violet',
    'OrderItem': 'cyan',
    'Review': 'coral'
}

# Draw subgraph
def draw_subgraph(center_id, depth):
    sub_meta = fan_out_graph_with_metadata(center_id, depth)
    sub_nodes = list(sub_meta.keys())
    subgraph = G.subgraph(sub_nodes).copy()

    node_colors = [color_map.get(subgraph.nodes[n].get('type', 'white'), 'white') for n in subgraph.nodes]

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(subgraph, k=0.5, iterations=30)
    nx.draw_networkx_nodes(subgraph, pos, node_size=300, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, arrows=True, arrowstyle='->', arrowsize=10)
    nx.draw_networkx_labels(subgraph, pos, font_size=7)
    plt.title(f"Graph for {center_id} (Depth {depth})")
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

# Display
if selected_id:
    st.subheader("🔗 Connected Entities")

    result_meta = fan_out_graph_with_metadata(selected_id, depth)

    # Generate summary
    depth_summary = defaultdict(lambda: defaultdict(int))
    for node, meta in result_meta.items():
        if meta["depth"] == 0:
            continue
        node_type = G.nodes[node].get("type", "Unknown")
        depth_summary[meta["depth"]][node_type] += 1

    summary_lines = []
    for d in sorted(depth_summary):
        total = sum(depth_summary[d].values())
        parts = [f"{v} {k.lower() + ('s' if v > 1 else '')}" for k, v in depth_summary[d].items()]
        summary_lines.append(f"Found {total} connections at depth {d} ({', '.join(parts)})")

    st.markdown("**Summary**")
    for line in summary_lines:
        st.markdown(f"- {line}")

    # Build display table
    table_data = []
    for node, meta in result_meta.items():
        node_type = G.nodes[node].get("type", "Unknown")
        url = f"?{urlencode({'entity_id': node, 'type': node_type, 'depth': depth})}"
        link = f"<a href='{url}' target='_blank'>{node}</a>"
        table_data.append({
            "ID": link,
            "Type": node_type,
            "Depth": meta["depth"],
            "How it's connected": meta["reason"]
        })

    st.markdown("### 📋 Connection Table")
    st.write("Click an ID to explore that entity in a new tab.")
    st.write(pd.DataFrame(table_data).to_html(escape=False, index=False), unsafe_allow_html=True)

    draw_subgraph(selected_id, depth)

    st.markdown("### 🎨 Node Type Legend")
    legend_html = "".join([f"<div style='display:inline-block; margin-right:10px;'><span style='display:inline-block; width:15px; height:15px; background-color:{color}; border:1px solid #000;'></span> {label}</div>" for label, color in color_map.items()])
    st.markdown(legend_html, unsafe_allow_html=True)
