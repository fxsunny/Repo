
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
from urllib.parse import urlencode
from collections import defaultdict

# ---------------- Setup ----------------
st.set_page_config(layout="wide")
st.title("🕵️‍♀️ RIA – Risk Intelligence Analyst")

# ---------------- Load or Create Graph ----------------
@st.cache_data
def load_graph():
    G = nx.DiGraph()
    edges = [
        ("R001", "CP010", "reviews_product"),
        ("C003", "R001", "writes_review"),
        ("PS100", "CP010", "has_variant"),
        ("S002", "O500", "makes_offer"),
        ("O500", "CP010", "offers_product"),
        ("C003", "ORD01", "places_order")
    ]
    for src, tgt, rel in edges:
        G.add_edge(src, tgt, label=rel)
    return G

G = load_graph()

# ---------------- Node Metadata ----------------
node_types = {
    "R001": "Review",
    "CP010": "ChildProduct",
    "C003": "Customer",
    "PS100": "ProductSet",
    "S002": "Seller",
    "O500": "Offer",
    "ORD01": "Order"
}

# ---------------- Fan-Out Utility ----------------
def fan_out_with_depth(G, start, max_depth):
    visited = {start: {"depth": 0, "reason": "Origin"}}
    frontier = [start]
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for node in frontier:
            for neighbor in G.successors(node):
                if neighbor not in visited:
                    reason = f"{node} → {neighbor} via '{G.edges[node, neighbor]['label']}'"
                    visited[neighbor] = {"depth": depth, "reason": reason}
                    next_frontier.append(neighbor)
            for predecessor in G.predecessors(node):
                if predecessor not in visited:
                    reason = f"{predecessor} → {node} via '{G.edges[predecessor, node]['label']}'"
                    visited[predecessor] = {"depth": depth, "reason": reason}
                    next_frontier.append(predecessor)
        frontier = next_frontier
    return visited

# ---------------- UI Controls ----------------
entity_id = st.selectbox("🔍 Select an entity to investigate:", list(G.nodes))
depth = st.selectbox("📏 Select depth to explore:", [1, 2, 3, 4, 5], index=1)

selected_types = st.multiselect(
    "🧩 Filter by entity type:",
    options=sorted(set(node_types.values())),
    default=sorted(set(node_types.values()))
)

visited = fan_out_with_depth(G, entity_id, depth)

# ---------------- Summary ----------------
depth_summary = defaultdict(lambda: defaultdict(int))
for node_id, meta in visited.items():
    if meta["depth"] == 0:
        continue
    t = node_types.get(node_id, "Unknown")
    depth_summary[meta["depth"]][t] += 1

summary_lines = []
for d in sorted(depth_summary):
    total = sum(depth_summary[d].values())
    parts = [f"{v} {k.lower() + ('s' if v > 1 else '')}" for k, v in depth_summary[d].items()]
    line = f"🔎 Analyzing {total} connections at depth {d} ({', '.join(parts)})"
    summary_lines.append(line)

st.markdown("### 🧠 Summary")
for line in summary_lines:
    st.markdown(line)

# ---------------- Table ----------------
data = []
for node_id, meta in visited.items():
    t = node_types.get(node_id, "Unknown")
    if t in selected_types or node_id == entity_id:
        url = f"?entity={urlencode({'id': node_id})}"
        data.append({
            "Entity ID": f"[{node_id}]({url})",
            "Type": t,
            "Depth": meta["depth"],
            "Reason": meta["reason"]
        })

df = pd.DataFrame(data).sort_values(by=["Depth", "Type"])
st.markdown("### 📋 Connections Table")
st.write(df.to_html(escape=False), unsafe_allow_html=True)

# ---------------- Graphs by Depth ----------------
def draw_graph_at_depth(d):
    H = nx.DiGraph()
    for node_id, meta in visited.items():
        if meta["depth"] <= d:
            t = node_types.get(node_id, "Unknown")
            if t in selected_types or node_id == entity_id:
                H.add_node(node_id, type=t)
    for u, v in G.edges:
        if u in H.nodes and v in H.nodes:
            H.add_edge(u, v, label=G.edges[u, v]["label"])

    fig, ax = plt.subplots()
    pos = nx.spring_layout(H)
    labels = {n: n for n in H.nodes}
    nx.draw(H, pos, labels=labels, with_labels=True, node_size=1600, node_color="skyblue", font_size=10, ax=ax)
    edge_labels = nx.get_edge_attributes(H, 'label')
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=8)
    return fig

st.markdown("### 🌐 Visualizations by Depth")
for d in range(1, depth + 1):
    st.subheader(f"Depth {d}")
    fig = draw_graph_at_depth(d)
    st.pyplot(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label=f"📥 Export Depth {d} Graph as PNG",
        data=buf.getvalue(),
        file_name=f"graph_depth_{d}.png",
        mime="image/png"
    )
