import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from urllib.parse import urlencode
from io import BytesIO
from collections import defaultdict
from scoring import score_entity


# st.set_page_config(layout='wide')
st.title('RIA: Risk Intelligence Analyst')

# Load Excel data
df = pd.read_excel('Ecommerce_Graph_Data.xlsx', sheet_name=None)
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
    # Add nodes
    for _, row in brands.iterrows():
        G.add_node(row['BrandID'], type='Brand')
    for _, row in product_sets.iterrows():
        G.add_node(row['ProductSetID'], type='ProductSet')
    for _, row in child_products.iterrows():
        G.add_node(row['ChildProductID'], type='ChildProduct')
    for _, row in sellers.iterrows():
        G.add_node(row['SellerID'], type='Seller')
    for _, row in offers.iterrows():
        G.add_node(row['OfferID'], type='Offer')
    for _, row in customers.iterrows():
        G.add_node(row['CustomerID'], type='Customer')
    for _, row in orders.iterrows():
        G.add_node(row['OrderID'], type='Order')
    for _, row in order_items.iterrows():
        G.add_node(row['OrderItemID'], type='OrderItem')
    for _, row in reviews.iterrows():
        G.add_node(row['ReviewID'], type='Review')

    # Add edges
    for _, row in product_sets.iterrows():
        G.add_edge(row['BrandID'], row['ProductSetID'], relation='owns')
    for _, row in child_products.iterrows():
        G.add_edge(row['ProductSetID'], row['ChildProductID'], relation='has_variant')
    for _, row in offers.iterrows():
        G.add_edge(row['SellerID'], row['OfferID'], relation='makes_offer')
        G.add_edge(row['OfferID'], row['ChildProductID'], relation='offers_product')
    for _, row in orders.iterrows():
        G.add_edge(row['CustomerID'], row['OrderID'], relation='places_order')
    for _, row in order_items.iterrows():
        G.add_edge(row['OrderID'], row['OrderItemID'], relation='contains_item')
        G.add_edge(row['OrderItemID'], row['OfferID'], relation='item_offer')
    for _, row in reviews.iterrows():
        G.add_edge(row['CustomerID'], row['ReviewID'], relation='writes_review')
        G.add_edge(row['ReviewID'], row['ChildProductID'], relation='reviews_product')

build_graph()

# Parse query parameters
params = st.query_params
selected_id = params.get('entity_id', None)
selected_type = params.get('type', None)
param_depth = params.get('depth')

# Determine initial default depth from query param or fallback
default_depth = int(param_depth) if param_depth else 4

# Sidebar settings
depth = st.sidebar.selectbox("Connection Depth", [1, 2, 3, 4, 5], index=default_depth - 1)
show_labels = st.sidebar.checkbox("Display Labels", value=True)

# Entity types for dropdown
entity_types = {
    'Brand': brands['BrandID'].tolist(),
    'Review': reviews['ReviewID'].tolist(),
    'Customer': customers['CustomerID'].tolist(),
    'Seller': sellers['SellerID'].tolist(),
    'ProductSet': product_sets['ProductSetID'].tolist(),
    'ChildProduct': child_products['ChildProductID'].tolist(),
    'Offer': offers['OfferID'].tolist(),
    'Order': orders['OrderID'].tolist(),
    'OrderItem': order_items['OrderItemID'].tolist()
}


# Fallback to manual dropdown if no query param
if not selected_id or not selected_type:
    selected_type = st.selectbox('Select Entity Type', list(entity_types.keys()))
    selected_id = st.selectbox(f'Select {selected_type} ID', entity_types[selected_type])

# Expand from selected entity using breadth-first search
def fan_out_graph(center_id, depth=1):
    visited = set([center_id])
    current_layer = [center_id]
    for _ in range(depth):
        next_layer = []
        for node in current_layer:
            neighbors = list(G.successors(node)) + list(G.predecessors(node))
            for n in neighbors:
                if n not in visited:
                    next_layer.append(n)
        visited.update(next_layer)
        current_layer = next_layer
    return visited

# Graph coloring
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

def draw_subgraph(center_id, depth, show_labels=True):
    sub_nodes = fan_out_graph(center_id, depth)
    subgraph = G.subgraph(sub_nodes).copy()

    node_colors = [
        color_map.get(subgraph.nodes[n].get('type', 'Offer'), 'white') for n in subgraph.nodes
    ]

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(subgraph, k=0.5, iterations=30)
    nx.draw_networkx_nodes(subgraph, pos, node_size=300, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, arrows=True, arrowstyle='->', arrowsize=10)
    nx.draw_networkx_labels(subgraph, pos, font_size=7)
    plt.title(f'Graph for {center_id} (Depth {depth})')
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

# Compute visited nodes using breadth-first search up to selected depth
def compute_visited_nodes(center_id, depth):
    visited = {center_id: {'depth': 0, 'reason': 'Origin'}}
    current_layer = [center_id]
    for d in range(1, depth + 1):
        next_layer = []
        for node in current_layer:
            neighbors = list(G.successors(node)) + list(G.predecessors(node))
            for n in neighbors:
                if n not in visited:
                    # Get edge relation/label
                    relation = 'connected'
                    if G.has_edge(node, n):
                        relation = G.edges[node, n].get('relation', 'connected')
                    elif G.has_edge(n, node):
                        relation = G.edges[n, node].get('relation', 'connected')
                    
                    reason = f'{node} ⇄ {n} via \'{relation}\''
                    visited[n] = {'depth': d, 'reason': reason}
                    next_layer.append(n)
        current_layer = next_layer
    return visited

# Run and display
if selected_id:
    st.subheader('Connected Entities')
    connected_nodes = fan_out_graph(selected_id, depth)
    # st.markdown(f'{len(connected_nodes)} nodes connected to `{selected_id}` within {depth} hops.')
    
    # Compute visited nodes for depth visualizations
    visited = compute_visited_nodes(selected_id, depth)

    # Compute and display risk score
    score_result = score_entity(selected_id, visited, G)
    st.markdown(f"### 🧮 Risk Score: **{score_result['risk_score']}**")
    if score_result['abuse_tags']:
        st.write("🚩 Abuse Tags:", ", ".join(score_result['abuse_tags']))
    st.caption(f"ℹ️ Reason: {score_result['reason']}")

    # Create summary per depth and type    
    depth_summary = defaultdict(lambda: defaultdict(int))
    for node, meta in visited.items():
        if node == selected_id:
            continue
        d = meta.get('depth', '?')
        t = G.nodes[node].get('type', 'Unknown')
        depth_summary[d][t] += 1
    
    # Build and display summary string
    summary_lines = [f"Analyzing {len(visited)-1} connections from `{selected_id}`:"]
    for d in range(1, depth + 1):
        type_counts = depth_summary.get(d, {})
        type_parts = [f"{v} {k.lower() + ('s' if v > 1 else '')}" for k, v in sorted(type_counts.items())]
        if type_parts:
            summary_lines.append(f"- Within {d} hop{'s' if d > 1 else ''}: " + ", ".join(type_parts))
        else:
            summary_lines.append(f"- Within {d} hop{'s' if d > 1 else ''}: No connections")
    
    st.markdown("  \n".join(summary_lines))

    
    # Prepare detailed connected entity table
    if visited:
        rows = []
        for node, meta in visited.items():
            if node == selected_id:
                continue
            node_type = G.nodes[node].get('type', 'Unknown')
            d = meta.get('depth', '?')
            reason = meta.get('reason', '')
            params_dict = {'entity_id': node, 'type': node_type, 'depth': depth}
            params_str = urlencode(params_dict)
            hyperlink = f'<a href="?{params_str}" target="_blank">{node}</a>'
            rows.append({'Entity ID': hyperlink, 'Type': node_type, 'Depth': d, 'Connection Description': reason})
    
        df_display = pd.DataFrame(rows)
        st.markdown('### 🔗 Connected Entities')
        st.write(f"Total connected entities: {len(df_display)}")
        st.write("Click any ID to explore it in a new tab.")
        st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.warning("No connected entities found.")
    
    # Draw main subgraph
    draw_subgraph(selected_id, depth, show_labels)

    # Multiple Graph Visualizations by Depth
    st.markdown('### 🌐 Visualizations by Depth')
    for d in range(1, depth + 1):
        subgraph_nodes = [n for n, meta in visited.items() if meta['depth'] <= d]
        H = G.subgraph(subgraph_nodes).copy()

        if len(H.nodes()) > 0:  # Only create visualization if there are nodes
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(H, k=0.5, iterations=30)
            
            # Color nodes by type
            node_colors = [color_map.get(H.nodes[n].get('type', 'Unknown'), 'white') for n in H.nodes()]
            
            nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=500, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle='->', arrowsize=15, ax=ax)
            nx.draw_networkx_labels(H, pos, font_size=7, ax=ax)
            
            # Add edge labels if they exist
            edge_labels = {}
            for edge in H.edges():
                relation = H.edges[edge].get('relation', '')
                if relation:
                    edge_labels[edge] = relation
            
            if edge_labels:
                nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=6, ax=ax)

            ax.set_title(f'Depth {d} - {len(H.nodes())} nodes, {len(H.edges())} edges')
            ax.axis('off')
            
            st.subheader(f'Depth {d}')
            st.pyplot(fig)
            plt.close(fig)

            # Create legend
            legend_text = '🎨 **Node Colors:** '
            used_types = set(H.nodes[n].get('type', 'Unknown') for n in H.nodes())
            legend_items = []
            for node_type in used_types:
                color = color_map.get(node_type, 'white')
                legend_items.append(f'{color.title()} = {node_type}')
            st.caption(legend_text + ' | '.join(legend_items))

            # PNG export
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label=f'📥 Export Depth {d} Graph as PNG',
                data=buf.getvalue(),
                file_name=f'graph_depth_{d}_{selected_id}.png',
                mime='image/png'
            )
        else:
            st.write(f'No nodes found at depth {d}')
