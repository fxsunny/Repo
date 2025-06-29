import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import io
import random
from typing import Dict, List, Tuple, Any, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output
from IPython import get_ipython


class DataLoader:
    """Handles data loading and generation operations"""
    
    @staticmethod
    def generate_sample_data_with_rings(excel_file=None) -> Dict:
        if excel_file:
            return DataLoader._load_from_excel(excel_file)
        return DataLoader._generate_synthetic_data()
    
    @staticmethod
    def _load_from_excel(excel_file) -> Dict:
        xls = pd.ExcelFile(excel_file)
        customers = xls.parse("Customers")["customer_id"].tolist()
        sellers = xls.parse("Sellers")["seller_id"].tolist()
        brand_df = xls.parse("Brands")
        brands = dict(zip(brand_df["brand_id"], brand_df["brand_name"]))
        prod_df = xls.parse("Products")
        products = {
            row["product_id"]: {"brand": row["brand"], "seller": row["seller"]}
            for _, row in prod_df.iterrows()
        }
        rev_df = xls.parse("Reviews")
        reviews = rev_df.to_dict("records")
        
        rings = []
        if "InjectedRings" in xls.sheet_names:
            ring_df = xls.parse("InjectedRings")
            for ring_id, group in ring_df.groupby("ring_id"):
                customers_in_ring = group["customer_id"].unique().tolist()
                products_in_ring = group["product_id"].unique().tolist()
                rings.append((customers_in_ring, products_in_ring))
        
        return {
            "customers": customers,
            "sellers": sellers,
            "brands": brands,
            "products": products,
            "reviews": reviews,
            "rings": rings,
            "source": excel_file
        }
    
    @staticmethod
    def _generate_synthetic_data() -> Dict:
        random.seed(42)
        num_customers = 120
        num_brands = 5
        num_sellers = 4
        num_products_per_brand = 8
        num_random_reviews = 500
        ring_configs = [(5, 3), (4, 4), (6, 5), (3, 2), (7, 3)]
        
        customers = [f"C{str(i).zfill(3)}" for i in range(1, num_customers + 1)]
        brands = {f"B{str(i).zfill(2)}": f"Brand{i}" for i in range(1, num_brands + 1)}
        sellers = [f"S{str(i).zfill(2)}" for i in range(1, num_sellers + 1)]
        
        # Generate products
        products = {}
        for b in brands:
            for i in range(1, num_products_per_brand + 1):
                pid = f"{b}P{str(i).zfill(3)}"
                products[pid] = {"brand": b, "seller": random.choice(sellers)}
        
        # Generate reviews and rings
        reviews = []
        review_id = 1
        
        # Random reviews
        for _ in range(num_random_reviews):
            reviews.append({
                "review_id": f"R{str(review_id).zfill(3)}",
                "customer": random.choice(customers),
                "product": random.choice(list(products.keys())),
                "rating": random.randint(1, 5)
            })
            review_id += 1
        
        # Generate ring reviews
        ring_customers = customers[:sum(c for c, _ in ring_configs)]
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


        # Stub: Self-Review injection
        # Placeholder: Inject reviews from seller IDs also acting as customers
        # Example:
        # for seller_id in sellers[:1]:
        #     reviews.append({"review_id": f"R{str(review_id).zfill(3)}", "customer": seller_id, "product": ..., "rating": 5})
        #     review_id += 1
        
        # Stub: Review Swap injection
        # Placeholder: Create interlinked dummy reviews across two seller product groups
        
        # Stub: Review Bursts injection
        # Placeholder: Assign identical or near-identical timestamps to many reviews on a product
        
        # Stub: Product Hijacking injection
        # Placeholder: Simulate product lineage change and suspicious review carryovers
        
        # Stub: Return Abuse / Brushing injection
        # Placeholder: Add dummy orders with no shipment but linked reviews
        
        # Stub: Sockpuppet injection
        # Placeholder: Add a customer that reviews 20+ products from 1 seller
        # reviews = inject_self_reviews(reviews, products)
        # reviews = inject_review_swap_rings(reviews, swap_pairs=[("S_1", "S_2")], products=products)
        # reviews = inject_review_bursts(reviews, product_id="P_3")
        # reviews = inject_product_hijack(reviews, hijacked_product="P_hijacked", donor_reviews=reviews)
        # reviews = inject_brushing_orders(reviews)
        # reviews = inject_sockpuppet_behavior(reviews, seller="S_3", products=["P_9", "P_10", "P_11"])

        
        return {
            "customers": customers,
            "sellers": sellers,
            "brands": brands,
            "products": products,
            "reviews": reviews,
            "rings": ring_segments,
            "source": "generated"
        }


class DataExporter:
    """Handles data export operations"""
    
    @staticmethod
    def save_to_excel(data: Dict, buffer) -> None:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            pd.DataFrame({"customer_id": data["customers"]}).to_excel(
                writer, sheet_name="Customers", index=False)
            pd.DataFrame({"seller_id": data["sellers"]}).to_excel(
                writer, sheet_name="Sellers", index=False)
            pd.DataFrame([
                {"brand_id": b_id, "brand_name": b_name}
                for b_id, b_name in data["brands"].items()
            ]).to_excel(writer, sheet_name="Brands", index=False)
            pd.DataFrame([
                {"product_id": pid, "brand": meta["brand"], "seller": meta["seller"]}
                for pid, meta in data["products"].items()
            ]).to_excel(writer, sheet_name="Products", index=False)
            pd.DataFrame(data["reviews"]).to_excel(
                writer, sheet_name="Reviews", index=False)
            
            if data.get("rings"):
                ring_rows = []
                for i, (custs, prods) in enumerate(data["rings"], 1):
                    for c in custs:
                        for p in prods:
                            ring_rows.append({
                                "ring_id": i,
                                "customer_id": c,
                                "product_id": p
                            })
                pd.DataFrame(ring_rows).to_excel(
                    writer, sheet_name="InjectedRings", index=False)
    
    @staticmethod
    def export_ring_clusters_to_csv(rings: List[Dict], customer_product: Dict, 
                                  rating_lookup: Dict) -> pd.DataFrame:
        download_rows = []
        for idx, ring in enumerate(rings, 1):
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
        return pd.DataFrame(download_rows)




class GraphBuilder:
    """Handles graph construction and analysis"""
    
    @staticmethod
    def build_graph(data: Dict) -> nx.DiGraph:
        G = nx.DiGraph()
        
        # Add nodes
        for c in data["customers"]:
            G.add_node(c, label="customer")
            
        for s in data["sellers"]:
            G.add_node(s, label="seller")
            
        for b_id, b_name in data["brands"].items():
            G.add_node(b_id, label="brand", name=b_name)
            
        for p_id, meta in data["products"].items():
            G.add_node(p_id, label="product")
            G.add_edge(p_id, meta["brand"], type="belongs_to")
            G.add_edge(p_id, meta["seller"], type="sold_by")
            
        for r in data["reviews"]:
            r_id = r["review_id"]
            G.add_node(r_id, label="review", rating=r["rating"])
            G.add_edge(r["customer"], r_id, type="wrote")
            G.add_edge(r_id, r["product"], type="about")
            
        return G
    
    @staticmethod
    def build_projection_graph(reviews: List[Dict], min_shared: int) -> Tuple[nx.Graph, Dict]:
        customer_product = {}
        for r in reviews:
            c, p = r["customer"], r["product"]
            customer_product.setdefault(c, set()).add(p)
            
        G_proj = nx.Graph()
        
        for c1 in customer_product:
            for c2 in customer_product:
                if c1 >= c2:
                    continue
                shared = customer_product[c1].intersection(customer_product[c2])
                if len(shared) >= min_shared:
                    G_proj.add_edge(c1, c2, weight=len(shared))
                    
        return G_proj, customer_product
    
    @staticmethod
    def export_graph_visualization(G: nx.Graph, filename: str, format: str = "png") -> bytes:
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_color="skyblue", 
                node_size=1000, font_size=10, edge_color="gray", ax=ax)
        plt.title("Customer-Shared Review Product Graph (Projection)", fontsize=12)
        
        buf = io.BytesIO()
        plt.savefig(buf, format=format)
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

class RingDetector:
    """Handles review ring detection and scoring"""
    
    def __init__(self, data: Dict):
        self.data = data
        self.products = data["products"]
        self.brands = data["brands"]
        self.reviews = data["reviews"]
        
    def detect_rings(self, G_proj: nx.Graph, customer_product: Dict) -> List[Dict]:
        detected_rings_raw = list(nx.connected_components(G_proj))
        suspicious_rings = []
        
        rating_lookup = {
            (r["customer"], r["product"]): r["rating"] 
            for r in self.reviews
        }
        
        for ring in detected_rings_raw:
            ring_data = self._analyze_ring(ring, customer_product)
            suspicious_rings.append(ring_data)
            
        suspicious_rings.sort(key=lambda r: r["suspicion_score"], reverse=True)
        return suspicious_rings, rating_lookup
    
    def _analyze_ring(self, ring: set, customer_product: Dict) -> Dict:
        ring_customers = list(ring)
        reviewed_products = set()
        
        for cust in ring_customers:
            reviewed_products |= customer_product[cust]
            
        involved_sellers = {
            self.products[p]["seller"] 
            for p in reviewed_products 
            if p in self.products
        }
        
        involved_brands = {
            self.products[p]["brand"] 
            for p in reviewed_products 
            if p in self.products
        }
        
        score = self._calculate_suspicion_score(
            len(ring_customers),
            len(reviewed_products),
            len(involved_sellers),
            len(involved_brands)
        )
        
        return {
            "customers": ring_customers,
            "shared_products": list(reviewed_products),
            "sellers": list(involved_sellers),
            "brands": [self.brands[b] for b in involved_brands],
            "suspicion_score": round(score, 1)
        }
    
    def _calculate_suspicion_score(
        self, n_customers: int, n_products: int, 
        n_sellers: int, n_brands: int
    ) -> float:
        seller_score = 30 * (1 if n_sellers == 1 else 0.5 if n_sellers == 2 else 0)
        brand_score = 30 * (1 if n_brands == 1 else 0.5 if n_brands == 2 else 0)
        product_score = 20 * min(n_products / 5, 1)
        group_score = 20 * min(n_customers / 5, 1)
        
        return seller_score + brand_score + product_score + group_score



class GraphVisualizer:
    """Handles all visualization related functionality"""
    
    def __init__(self):
        self.color_map = {
            "customer": "lightblue",
            "review": "orange",
            "product": "lightgreen",
            "seller": "pink",
            "brand": "violet"
        }
        
    def plot_full_graph(self, G: nx.DiGraph) -> plt.Figure:
        node_colors = [
            self.color_map.get(G.nodes[n].get("label", "unknown"), "gray") 
            for n in G.nodes
        ]
        
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
        
        edge_labels = nx.get_edge_attributes(G, "type")
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=7,
            ax=ax
        )
        
        plt.title("Multi-Entity Review Graph", fontsize=14)
        return fig
    
    def plot_ring_subgraph(
        self, 
        subgraph: nx.Graph,
        customer_product: Dict,
        products: Dict,
        brands: Dict,
        ring_number: int,
        brand_color_map: Dict
    ) -> plt.Figure:
        cust_colors = []
        for cust in subgraph.nodes():
            reviewed = customer_product[cust]
            brands_reviewed = {
                products[p]["brand"] 
                for p in reviewed 
                if p in products
            }
            dominant_brand = next(
                (b for b in brands_reviewed if b in brands), 
                None
            )
            brand_name = brands.get(dominant_brand, "Unknown")
            cust_colors.append(brand_color_map.get(brand_name, "gray"))
            
        fig, ax = plt.subplots(figsize=(6, 4))
        nx.draw(
            subgraph,
            with_labels=True,
            node_color=cust_colors,
            node_size=1000,
            font_weight="bold",
            edge_color="gray",
            ax=ax
        )
        
        plt.title(f"Review Ring {ring_number}: Customer-Customer Graph (Colored by Brand)")
        return fig
    
    @staticmethod
    def create_brand_legend(brand_color_map: Dict) -> plt.Figure:
        legend_items = [
            mpatches.Patch(color=color, label=brand) 
            for brand, color in brand_color_map.items()
        ]
        
        fig, ax = plt.subplots()
        ax.legend(handles=legend_items, loc="center", ncol=3, frameon=False)
        ax.axis("off")
        return fig

class StreamlitUI:
    """Handles Streamlit UI components and layout"""
    
    def __init__(self):
        self.visualizer = GraphVisualizer()
        self.data_exporter = DataExporter()
        
    def render_header(self) -> None:
        st.title("🛒 E-Commerce Abuse Graph: Multi-Entity Review Network")
        st.markdown("This network graph includes customers, reviews, products, sellers, and brands.")
        
    def render_sidebar(self, data: Dict) -> None:
        if "xlsx" in data.get("source", ""):
            st.sidebar.success("📂 Data loaded from Excel")
        else:
            st.sidebar.info("🧪 Using synthetic sample data")
        
        # Preview injected rings in sidebar
        st.sidebar.markdown("### 🔍 Injected Review Rings")
        for i, (ring_custs, ring_prods) in enumerate(data["rings"], 1):
            st.sidebar.write(f"Ring {i}: {len(ring_custs)} customers, {len(ring_prods)} products")
        
        # Add download buttons to sidebar
        self.render_download_buttons(data)
        
    def render_download_buttons(self, data: Dict) -> None:
        if st.sidebar.button("📥 Download Sample Dataset as Excel"):
            excel_buffer = io.BytesIO()
            self.data_exporter.save_to_excel(data, excel_buffer)
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Download Excel",
                data=excel_buffer,
                file_name="sample_review_graph_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    def render_data_preview(self, data: Dict) -> None:
        with st.expander("📄 Preview Sample Reviews", expanded=False):
            st.dataframe(pd.DataFrame(data["reviews"][:20]))
    
    def render_ring_details(
        self,
        ring: Dict,
        index: int,
        G_proj: nx.Graph,
        customer_product: Dict,
        products: Dict,
        brands: Dict,
        brand_color_map: Dict
    ) -> None:
        st.markdown(f"### 🔸 Ring {index} — Suspicion Score: {ring['suspicion_score']} / 100")
        st.write(f"**Customers:** {', '.join(ring['customers'])}")
        st.write(f"**Shared Products Reviewed:** {', '.join(ring['shared_products'])}")
        st.write(f"**Sellers Involved:** {', '.join(ring['sellers'])}")
        st.write(f"**Brands Involved:** {', '.join(ring['brands'])}")
        
        # Explainability
        explanation = self._generate_explanation(ring)
        st.markdown(explanation)
        
        # Visualization
        subgraph = G_proj.subgraph(ring['customers'])
        fig = self.visualizer.plot_ring_subgraph(
            subgraph,
            customer_product,
            products,
            brands,
            index,
            brand_color_map
        )
        st.pyplot(fig)
        
    def render_validation_results(self, validation_df: pd.DataFrame) -> None:
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
        
        # Add download button for validation report
        if st.button("📥 Download Detection Report as Excel"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                validation_df.to_excel(writer, sheet_name="DetectionReport", index=False)
            buffer.seek(0)
            st.download_button(
                label="📊 Download Detection Report",
                data=buffer,
                file_name="detection_validation_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    def _generate_explanation(self, ring: Dict) -> str:
        explanation = (
            f"🔍 This ring includes {len(ring['customers'])} customers who "
            f"all reviewed {len(ring['shared_products'])} shared products."
        )
        
        if len(ring['sellers']) == 1 and len(ring['brands']) == 1:
            explanation += (" All of these products were sold by **1 seller** and "
                          "belonged to **1 brand**, suggesting potential coordinated "
                          "review activity.")
        elif len(ring['sellers']) <= 2 and len(ring['brands']) <= 2:
            explanation += (f" Most products are tied to just {len(ring['sellers'])} "
                          f"sellers and {len(ring['brands'])} brands, indicating "
                          "suspicious overlap.")
        else:
            explanation += (" The group spans multiple sellers and brands, so "
                          "further analysis is advised.")
            
        return explanation

    def render_export_options(self, G_proj: nx.Graph, suspicious_rings: List[Dict],
                            customer_product: Dict, rating_lookup: Dict) -> None:
        # Export ring clusters as CSV
        if suspicious_rings:
            df_download = self.data_exporter.export_ring_clusters_to_csv(
                suspicious_rings, customer_product, rating_lookup)
            st.download_button(
                "📥 Download Ring Cluster Data as CSV",
                data=df_download.to_csv(index=False),
                file_name="review_rings.csv",
                mime="text/csv"
            )
        
        # Export graph visualization
        st.subheader("📷 Export Graph Visualization")
        graph_bytes = GraphBuilder.export_graph_visualization(G_proj, "customer_review_graph")
        st.download_button(
            label="🖼️ Download Customer Graph as PNG",
            data=graph_bytes,
            file_name="customer_review_graph.png",
            mime="image/png"
        )

class JupyterUI:
    """Handles Jupyter Notebook UI components and layout"""
    
    def __init__(self):
        self.visualizer = GraphVisualizer()
        self.output = widgets.Output()
        self.min_shared_slider = widgets.IntSlider(
            value=2,
            min=1,
            max=5,
            step=1,
            description='Min Shared:',
            continuous_update=False
        )
        
    def render_header(self) -> None:
        display(widgets.HTML(
            "<h1>🛒 E-Commerce Abuse Graph: Multi-Entity Review Network</h1>"
            "<p>This network graph includes customers, reviews, products, sellers, and brands.</p>"
        ))
        
    def render_data_source(self, data_source: str) -> None:
        if "xlsx" in data_source:
            display(widgets.HTML("<div style='color: green'>📂 Data loaded from Excel</div>"))
        else:
            display(widgets.HTML("<div style='color: blue'>🧪 Using synthetic sample data</div>"))
            
    def get_min_shared_input(self, callback: Callable) -> None:
        """Sets up the minimum shared products slider with callback"""
        self.min_shared_slider.observe(callback, names='value')
        display(self.min_shared_slider)
        
    def render_ring_details(
        self,
        ring: Dict,
        index: int,
        G_proj: nx.Graph,
        customer_product: Dict,
        products: Dict,
        brands: Dict,
        brand_color_map: Dict
    ) -> None:
        with self.output:
            clear_output(wait=True)
            display(widgets.HTML(
                f"<h3>🔸 Ring {index} — Suspicion Score: {ring['suspicion_score']} / 100</h3>"
                f"<p><b>Customers:</b> {', '.join(ring['customers'])}</p>"
                f"<p><b>Shared Products:</b> {', '.join(ring['shared_products'])}</p>"
                f"<p><b>Sellers:</b> {', '.join(ring['sellers'])}</p>"
                f"<p><b>Brands:</b> {', '.join(ring['brands'])}</p>"
            ))
            
            # Explainability
            explanation = self._generate_explanation(ring)
            display(widgets.HTML(f"<p>{explanation}</p>"))
            
            # Visualization
            subgraph = G_proj.subgraph(ring['customers'])
            fig = self.visualizer.plot_ring_subgraph(
                subgraph,
                customer_product,
                products,
                brands,
                index,
                brand_color_map
            )
            display(fig)
            
    def render_validation_results(self, validation_df: pd.DataFrame) -> None:
        with self.output:
            display(widgets.HTML("<h2>✅ Detection Validation Report</h2>"))
            display(validation_df.style.apply(self._highlight_detection_status, axis=1))
            
    def _highlight_detection_status(self, row):
        highlight_color_map = {
            "✅ Exact Match": "background-color: #d4edda",
            "🟡 Partial (Subset)": "background-color: #fff3cd",
            "🟠 Partial (Superset)": "background-color: #ffeeba",
            "❌ Missed": "background-color: #f8d7da"
        }
        return [highlight_color_map.get(row["Detection Status"], "")] * len(row)
        
    def _generate_explanation(self, ring: Dict) -> str:
        explanation = (
            f"🔍 This ring includes {len(ring['customers'])} customers who "
            f"all reviewed {len(ring['shared_products'])} shared products."
        )
        
        if len(ring['sellers']) == 1 and len(ring['brands']) == 1:
            explanation += (" All of these products were sold by **1 seller** and "
                          "belonged to **1 brand**, suggesting potential coordinated "
                          "review activity.")
        elif len(ring['sellers']) <= 2 and len(ring['brands']) <= 2:
            explanation += (f" Most products are tied to just {len(ring['sellers'])} "
                          f"sellers and {len(ring['brands'])} brands, indicating "
                          "suspicious overlap.")
        else:
            explanation += (" The group spans multiple sellers and brands, so "
                          "further analysis is advised.")
            
        return explanation



class ReviewRingDetectionApp:
    """Main application class for Streamlit implementation"""
    
    def __init__(self):
        self.ui = StreamlitUI()
        self.data_loader = DataLoader()
        self.data_exporter = DataExporter()
        
    def run(self):
        self.ui.render_header()
        
        # Load or generate data
        data = self.data_loader.generate_sample_data_with_rings()
        self.ui.render_sidebar(data)
        self.ui.render_data_preview(data)
        
        # Build graphs
        G = GraphBuilder.build_graph(data)
        
        # Display main graph
        fig = self.ui.visualizer.plot_full_graph(G)
        st.pyplot(fig)
        
        # Ring detection
        min_shared = st.slider(
            "🔧 Minimum Shared Products for Strong Connection",
            1, 5, 2
        )
        G_proj, customer_product = GraphBuilder.build_projection_graph(
            data["reviews"],
            min_shared
        )
        
        detector = RingDetector(data)
        suspicious_rings, rating_lookup = detector.detect_rings(G_proj, customer_product)
        
        if suspicious_rings:
            st.success(f"Found {len(suspicious_rings)} potential review ring(s):")
            
            # Create brand color mapping
            brand_names = sorted(list({b for ring in suspicious_rings for b in ring["brands"]}))
            brand_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
            brand_color_map = {
                brand: brand_palette[i % len(brand_palette)] 
                for i, brand in enumerate(brand_names)
            }
            
            # Display ring details
            for i, ring in enumerate(suspicious_rings, 1):
                self.ui.render_ring_details(
                    ring,
                    i,
                    G_proj,
                    customer_product,
                    data["products"],
                    data["brands"],
                    brand_color_map
                )
            
            # Brand color legend
            st.subheader("🎨 Brand Color Legend")
            fig = self.ui.visualizer.create_brand_legend(brand_color_map)
            st.pyplot(fig)
            
            # Full graph visualization
            st.subheader("🌐 Full Customer Review Graph")
            fig_full, ax = plt.subplots(figsize=(8, 6))
            nx.draw(G_proj, with_labels=True, node_color="skyblue", 
                   node_size=1000, font_size=10, edge_color="gray", ax=ax)
            plt.title("Customer-Customer Graph (Shared Product Reviews)", fontsize=12)
            st.pyplot(fig_full)
            
            # Export options
            self.ui.render_export_options(G_proj, suspicious_rings, customer_product, rating_lookup)
            
            # Validation
            validation_df = ValidationManager.validate_detected_rings(
                data["rings"],
                suspicious_rings
            )
            self.ui.render_validation_results(validation_df)

class JupyterRingDetectionApp:
    """Main application class for Jupyter implementation"""
    
    def __init__(self):
        self.ui = JupyterUI()
        self.data_loader = DataLoader()
        self.data_exporter = DataExporter()
        self.data = None
        self.G = None
        self.G_proj = None
        self.customer_product = None
        
    def run(self):
        self.ui.render_header()
        
        # Load or generate data
        self.data = self.data_loader.generate_sample_data_with_rings()
        self.ui.render_data_source(self.data.get("source", "generated"))
        
        # Build main graph
        self.G = GraphBuilder.build_graph(self.data)
        
        # Display main graph
        fig = self.ui.visualizer.plot_full_graph(self.G)
        display(fig)
        
        # Set up slider with callback
        self.ui.get_min_shared_input(self._on_min_shared_change)
        display(self.ui.output)
        
    def _on_min_shared_change(self, change):
        """Callback for when minimum shared products slider changes"""
        min_shared = change['new']
        self.G_proj, self.customer_product = GraphBuilder.build_projection_graph(
            self.data["reviews"],
            min_shared
        )
        
        detector = RingDetector(self.data)
        suspicious_rings, rating_lookup = detector.detect_rings(self.G_proj, self.customer_product)
        
        if suspicious_rings:
            # Create brand color mapping
            brand_names = sorted(list({b for ring in suspicious_rings for b in ring["brands"]}))
            brand_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
            brand_color_map = {
                brand: brand_palette[i % len(brand_palette)] 
                for i, brand in enumerate(brand_names)
            }
            
            for i, ring in enumerate(suspicious_rings, 1):
                self.ui.render_ring_details(
                    ring,
                    i,
                    self.G_proj,
                    self.customer_product,
                    self.data["products"],
                    self.data["brands"],
                    brand_color_map
                )
            
            # Validation
            validation_df = ValidationManager.validate_detected_rings(
                self.data["rings"],
                suspicious_rings
            )
            self.ui.render_validation_results(validation_df)


class ValidationManager:
    """Handles validation of detected rings against injected rings"""
    
    @staticmethod
    def validate_detected_rings(
        injected_rings: List[Tuple], 
        detected_rings: List[Dict]
    ) -> pd.DataFrame:
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



# Environment detection and main execution
def is_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook/lab
            return True
        elif shell == 'TerminalInteractiveShell':  # IPython
            return True
        return False
    except NameError:
        return False

if __name__ == "__main__":
    if is_jupyter():
        app = JupyterRingDetectionApp()
    else:
        app = ReviewRingDetectionApp()  # Streamlit version
    app.run()
