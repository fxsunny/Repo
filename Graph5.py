import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import io
import random

import random

def generate_sample_data_with_rings():
    # Settings
    random.seed(42)
    num_customers = 120
    num_brands = 5
    num_sellers = 4
    num_products_per_brand = 8
    num_random_reviews = 500
    ring_configs = [   # ring sizes
        (5, 3),   # 5 customers review 3 shared products
        (4, 4),
        (6, 5),
        (3, 2),
        (7, 3)
    ]

    # IDs
    customers = [f"C{str(i).zfill(3)}" for i in range(1, num_customers + 1)]
    brands = {f"B{str(i).zfill(2)}": f"Brand{i}" for i in range(1, num_brands + 1)}
    sellers = [f"S{str(i).zfill(2)}" for i in range(1, num_sellers + 1)]

    # Products with brand prefix
    products = {}
    for b in brands.keys():
        for i in range(1, num_products_per_brand + 1):
            p_id = f"{b}P{str(i).zfill(3)}"
            products[p_id] = {
                "brand": b,
                "seller": random.choice(sellers)
            }

    # Generate random reviews
    reviews = []
    review_id = 1
    for _ in range(num_random_reviews):
        cust = random.choice(customers)
        prod = random.choice(list(products.keys()))
        reviews.append({
            "review_id": f"R{str(review_id).zfill(3)}",
            "customer": cust,
            "product": prod,
            "rating": random.randint(1, 5)
        })
        review_id += 1

    # Inject 5 review rings
    ring_customers = customers[:sum(c for c, _ in ring_configs)]  # Reserve from start
    ring_index = 0
    ring_segments = []
    for ring_size, shared_count in ring_configs:
        ring_custs = ring_customers[ring_index: ring_index + ring_size]
        ring_index += ring_size
        ring_prods = random.sample(list(products.keys()), shared_count)
        for cust in ring_custs:
            for prod in ring_prods:
                reviews.append({
                    "review_id": f"R{str(review_id).zfill(3)}",
                    "customer": cust,
                    "product": prod,
                    "rating": random.randint(4, 5)
                })
                review_id += 1
        ring_segments.append((ring_custs, ring_prods))

    return {
        "customers": customers,
        "sellers": sellers,
        "brands": brands,
        "products": products,
        "reviews": reviews,
        "rings": ring_segments  # Optional: use to test detection accuracy
    }

def validate_detected_rings(injected_rings, detected_rings):
    injected_sets = [set(custs) for custs, _ in injected_rings]
    detected_sets = [set(ring["customers"]) for ring in detected_rings]

    results = []
    for idx, inj in enumerate(injected_sets, 1):
        match_type = "❌ Missed"
        for det in detected_sets:
            if inj == det:
                match_type = "✅ Exact Match"
                break
            elif inj.issubset(det):
                match_type = "🟡 Partial (Subset)"
                break
            elif inj.issuperset(det):
                match_type = "🟠 Partial (Superset)"
                break

        results.append({
            "Injected Ring ID": idx,
            "Customers": ", ".join(sorted(injected_rings[idx - 1][0])),
            "Products": ", ".join(sorted(injected_rings[idx - 1][1])),
            "Detection Status": match_type
        })

    return pd.DataFrame(results)


# -------------------------
# 1. Sample Entity Data
# -------------------------
data = generate_sample_data_with_rings()
customers = data["customers"]
sellers = data["sellers"]
brands = data["brands"]
products = data["products"]
reviews = data["reviews"]
rings = data["rings"] 

# Optional: Preview injected rings in sidebar
st.sidebar.markdown("### 🔍 Injected Review Rings")
for i, (ring_custs, ring_prods) in enumerate(rings, 1):
    st.sidebar.write(f"Ring {i}: {len(ring_custs)} customers, {len(ring_prods)} products")

# Optional: Preview reviews
with st.expander("📄 Preview Sample Reviews", expanded=False):
    st.dataframe(pd.DataFrame(reviews[:20]))


# -------------------------
# 2. Build Graph
# -------------------------
G = nx.DiGraph()
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
st.title("\U0001F6D2 E-Commerce Abuse Graph: Multi-Entity Review Network")
st.markdown("This network graph includes customers, reviews, products, sellers, and brands.")

# -------------------------
# 4. Visualize Graph
# -------------------------
color_map = {"customer": "lightblue", "review": "orange", "product": "lightgreen", "seller": "pink", "brand": "violet"}
node_colors = [color_map.get(G.nodes[n].get("label", "unknown"), "gray") for n in G.nodes()]
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1000, font_size=8, font_weight='bold', edge_color="gray", ax=ax)
edge_labels = nx.get_edge_attributes(G, "type")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
plt.title("Multi-Entity Review Graph", fontsize=14)
st.pyplot(fig)

# -------------------------
# 5. Review Ring Detection
# -------------------------
st.header("\U0001F575\uFE0F Review Ring Detection")
customer_product = {}
for r in reviews:
    c, p = r["customer"], r["product"]
    customer_product.setdefault(c, set()).add(p)
min_shared = st.slider("\U0001F527 Minimum Shared Products for Strong Connection", 1, 5, 2)
G_proj = nx.Graph()
for c1 in customer_product:
    for c2 in customer_product:
        if c1 >= c2:
            continue
        shared = customer_product[c1].intersection(customer_product[c2])
        if len(shared) >= min_shared:
            G_proj.add_edge(c1, c2, weight=len(shared))
rings = list(nx.connected_components(G_proj))
suspicious_rings = []
rating_lookup = {(r["customer"], r["product"]): r["rating"] for r in reviews}

for ring in rings:
    ring_customers = list(ring)
    reviewed_products = set()
    for cust in ring_customers:
        reviewed_products |= customer_product[cust]
    involved_sellers = {products[p]["seller"] for p in reviewed_products if p in products}
    involved_brands = {products[p]["brand"] for p in reviewed_products if p in products}

    # Compute suspicion score
    n_customers = len(ring_customers)
    n_products = len(reviewed_products)
    n_sellers = len(involved_sellers)
    n_brands = len(involved_brands)
    seller_score = 30 * (1 if n_sellers == 1 else 0.5 if n_sellers == 2 else 0)
    brand_score = 30 * (1 if n_brands == 1 else 0.5 if n_brands == 2 else 0)
    product_score = 20 * min(n_products / 5, 1)
    group_score = 20 * min(n_customers / 5, 1)
    score = seller_score + brand_score + product_score + group_score

    suspicious_rings.append({
        "customers": ring_customers,
        "shared_products": list(reviewed_products),
        "sellers": list(involved_sellers),
        "brands": [brands[b] for b in involved_brands],
        "suspicion_score": round(score, 1)
    })

suspicious_rings.sort(key=lambda r: r["suspicion_score"], reverse=True)

if suspicious_rings:
    st.success(f"Found {len(suspicious_rings)} potential review ring(s):")
    brand_names = sorted(list({b for ring in suspicious_rings for b in ring["brands"]}))
    brand_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    brand_color_map = {brand: brand_palette[i % len(brand_palette)] for i, brand in enumerate(brand_names)}

    for i, ring in enumerate(suspicious_rings, 1):
        st.markdown(f"### 🔸 Ring {i} — Suspicion Score: {ring['suspicion_score']} / 100")
        st.write(f"**Customers:** {', '.join(ring['customers'])}")
        st.write(f"**Shared Products Reviewed:** {', '.join(ring['shared_products'])}")
        st.write(f"**Sellers Involved:** {', '.join(ring['sellers'])}")
        st.write(f"**Brands Involved:** {', '.join(ring['brands'])}")

        # Explainability
        explanation = f"🔍 This ring includes {len(ring['customers'])} customers who all reviewed {len(ring['shared_products'])} shared products."
        if len(ring['sellers']) == 1 and len(ring['brands']) == 1:
            explanation += " All of these products were sold by **1 seller** and belonged to **1 brand**, suggesting potential coordinated review activity."
        elif len(ring['sellers']) <= 2 and len(ring['brands']) <= 2:
            explanation += f" Most products are tied to just {len(ring['sellers'])} sellers and {len(ring['brands'])} brands, indicating suspicious overlap."
        else:
            explanation += " The group spans multiple sellers and brands, so further analysis is advised."
        st.markdown(explanation)

        # Subgraph for visualization
        subgraph = G_proj.subgraph(ring['customers'])
        cust_colors = []
        for cust in subgraph.nodes():
            reviewed = customer_product[cust]
            brands_reviewed = {products[p]["brand"] for p in reviewed if p in products}
            dominant_brand = next((b for b in brands_reviewed if b in brands), None)
            brand_name = brands.get(dominant_brand, "Unknown")
            cust_colors.append(brand_color_map.get(brand_name, "gray"))
        fig, ax = plt.subplots(figsize=(6, 4))
        nx.draw(subgraph, with_labels=True, node_color=cust_colors, node_size=1000, font_weight="bold", edge_color="gray", ax=ax)
        plt.title(f"Review Ring {i}: Customer-Customer Graph (Colored by Brand)")
        st.pyplot(fig)

    # Brand color legend
    st.subheader("🎨 Brand Color Legend")
    legend_items = [mpatches.Patch(color=color, label=brand) for brand, color in brand_color_map.items()]
    fig, ax = plt.subplots()
    ax.legend(handles=legend_items, loc="center", ncol=3, frameon=False)
    ax.axis("off")
    st.pyplot(fig)

    # Full graph visualization
    st.subheader("🌐 Full Customer Review Graph")
    fig_full, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G_proj, with_labels=True, node_color="skyblue", node_size=1000, font_size=10, edge_color="gray", ax=ax)
    plt.title("Customer-Customer Graph (Shared Product Reviews)", fontsize=12)
    st.pyplot(fig)

    # Step 6: Prepare and download CSV
    download_rows = []
    for idx, ring in enumerate(suspicious_rings, 1):
        for cust in ring["customers"]:
            for product in customer_product[cust]:
                rating = rating_lookup.get((cust, product), "")
                download_rows.append({
                    "Ring_ID": idx,
                    "Customer_ID": cust,
                    "Product_Reviewed": product,
                    "Rating": rating,
                    "Sellers": ", ".join(ring["sellers"]),
                    "Brands": ", ".join(ring["brands"]),
                    "Suspicion_Score": ring["suspicion_score"]
                })
    df_download = pd.DataFrame(download_rows)
    st.download_button("📥 Download Ring Cluster Data as CSV", data=df_download.to_csv(index=False), file_name="review_rings.csv", mime="text/csv")

# Step 7: Export graph as PNG
fig_full, ax = plt.subplots(figsize=(8, 6))
nx.draw(G_proj, with_labels=True, node_color="skyblue", node_size=1000, font_size=10, edge_color="gray", ax=ax)
plt.title("Customer-Shared Review Product Graph (Projection)", fontsize=12)
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
st.subheader("📷 Export Graph Visualization")
st.pyplot(fig_full)
st.download_button(label="🖼️ Download Customer Graph as PNG", data=buf, file_name="customer_review_graph.png", mime="image/png")

# -------------------------
# 8. Validation of Injected Rings
# -------------------------
validation_df = validate_detected_rings(rings, suspicious_rings)
st.subheader("✅ Detection Validation Report")
st.markdown("Compare injected rings vs detected ones:")

highlight_color_map = {
    "✅ Exact Match": "background-color: #d4edda",
    "🟡 Partial (Subset)": "background-color: #fff3cd",
    "🟠 Partial (Superset)": "background-color: #ffeeba",
    "❌ Missed": "background-color: #f8d7da"
}

def highlight_detection_status(row):
    return [highlight_color_map.get(row["Detection Status"], "")] * len(row)

st.dataframe(validation_df.style.apply(highlight_detection_status, axis=1))


