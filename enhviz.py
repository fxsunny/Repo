import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Any
import pandas as pd
import numpy as np

class EnhancedVisualization:
    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.color_map = {
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

    def _get_subgraph_nodes(self, center_id: str, depth: int):
        visited = set([center_id])
        current_layer = [center_id]
        for _ in range(depth):
            next_layer = []
            for node in current_layer:
                neighbors = list(self.G.successors(node)) + list(self.G.predecessors(node))
                for n in neighbors:
                    if n not in visited:
                        next_layer.append(n)
            visited.update(next_layer)
            current_layer = next_layer
        
        return self.G.subgraph(visited)
        
    def create_interactive_plot(self, center_id: str, depth: int):
        # Creates interactive plot with zoom/pan
        sub_nodes = self._get_subgraph_nodes(center_id, depth)
        pos = nx.spring_layout(sub_nodes)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add nodes
        node_trace = go.Scatter(
            x=[pos[node][0] for node in sub_nodes],
            y=[pos[node][1] for node in sub_nodes],
            mode='markers+text',
            hoverinfo='text',
            text=[f"ID: {node}<br>Type: {self.G.nodes[node]['type']}" for node in sub_nodes],
            marker=dict(
                size=20,
                color=[self.color_map[self.G.nodes[node]['type']] for node in sub_nodes]
            )
        )
        
        fig.add_trace(node_trace)
        
        # Add zoom/pan controls
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            dragmode='pan',
            clickmode='event+select'
        )
        
        return fig

    def _create_minimap(self):
        # Create a simplified version of the graph for the minimap
        mini_pos = nx.spring_layout(self.G)
        
        # Convert to base64 image string
        import io
        import base64
        from PIL import Image
        
        buf = io.BytesIO()
        plt.figure(figsize=(2,2))
        nx.draw(self.G, mini_pos, node_size=1)
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        # Convert to base64 string
        img_str = base64.b64encode(buf.getvalue()).decode()
        return f'data:image/png;base64,{img_str}'
    
    def add_minimap(self, fig: go.Figure):
        # Adds minimap to main visualization
        fig.add_layout_image(
            dict(
                source=self._create_minimap(),
                xref="paper",
                yref="paper",
                x=1,
                y=1,
                sizex=0.2,
                sizey=0.2,
                xanchor="right",
                yanchor="top"
            )
        )
        return fig


class AdvancedAnalytics:
    def __init__(self, G: nx.DiGraph, df_dict: Dict[str, pd.DataFrame]):
        self.G = G
        self.df_dict = df_dict

#    def _detect_unusual_activity(self, node_id: str) -> List[Dict[str, Any]]:
#        unusual_activities = []
#        node_type = self.G.nodes[node_id]['type']
#        # Add your unusual activity detection logic here
#        return unusual_activities
    
#    def _identify_clusters(self, node_id: str) -> List[Dict[str, Any]]:
#        clusters = []
#        # Add your clustering logic here
#        return clusters

    def _detect_unusual_activity(self, node_id: str) -> List[Dict[str, Any]]:
        unusual_activities = []
        node_type = self.G.nodes[node_id]['type']
        
        if node_type == 'Customer':
            # Get all reviews by this customer
            customer_reviews = self.df_dict['Reviews'][self.df_dict['Reviews']['CustomerID'] == node_id]
            
            # Check for rapid reviewing (multiple reviews in short time periods)
            if len(customer_reviews) > 0:
                review_dates = pd.to_datetime(customer_reviews['ReviewDate'])
                time_diffs = review_dates.diff()
                rapid_reviews = time_diffs[time_diffs.dt.total_seconds() < 3600]  # Reviews within 1 hour
                
                if not rapid_reviews.empty:
                    unusual_activities.append({
                        'type': 'rapid_reviewing',
                        'count': len(rapid_reviews),
                        'timestamps': rapid_reviews.index.tolist()
                    })
            
            # Check for extreme rating patterns
            if len(customer_reviews) >= 3:  # Minimum reviews threshold
                extreme_ratings = customer_reviews[
                    (customer_reviews['Rating'] == 1) | 
                    (customer_reviews['Rating'] == 5)
                ]
                extreme_ratio = len(extreme_ratings) / len(customer_reviews)
                
                if extreme_ratio > 0.8:  # 80% of reviews are extreme
                    unusual_activities.append({
                        'type': 'extreme_rating_bias',
                        'ratio': extreme_ratio,
                        'review_count': len(customer_reviews)
                    })
        
        return unusual_activities

    def _identify_clusters(self, node_id: str) -> List[Dict[str, Any]]:
        clusters = []
        node_type = self.G.nodes[node_id]['type']
        
        if node_type == 'Customer':
            # Get all reviews by this customer
            customer_reviews = self.df_dict['Reviews'][self.df_dict['Reviews']['CustomerID'] == node_id]
            
            if len(customer_reviews) > 0:
                # Cluster by product categories
                reviewed_products = customer_reviews['ChildProductID'].tolist()
                product_categories = []
                
                for cp_id in reviewed_products:
                    # Find ProductSetID for this ChildProduct
                    ps_id = self.df_dict['ChildProducts'][
                        self.df_dict['ChildProducts']['ChildProductID'] == cp_id
                    ]['ProductSetID'].values[0]
                    
                    # Get category for this ProductSet
                    category = self.df_dict['ProductSets'][
                        self.df_dict['ProductSets']['ProductSetID'] == ps_id
                    ]['Category'].values[0]
                    
                    product_categories.append(category)
                
                # Count reviews per category
                category_counts = pd.Series(product_categories).value_counts()
                
                # Identify dominant categories (>50% of reviews)
                for category, count in category_counts.items():
                    if count/len(customer_reviews) > 0.5:
                        clusters.append({
                            'type': 'category_focus',
                            'category': category,
                            'review_count': count,
                            'percentage': count/len(customer_reviews) * 100
                        })
                
                # Time-based clustering
                review_dates = pd.to_datetime(customer_reviews['ReviewDate'])
                time_clusters = pd.cut(review_dates, bins=5)  # Split into 5 time periods
                time_distribution = time_clusters.value_counts()
                
                # Check for time-based review clusters
                max_cluster_size = time_distribution.max()
                if max_cluster_size/len(customer_reviews) > 0.4:  # 40% of reviews in one time period
                    clusters.append({
                        'type': 'time_cluster',
                        'period': time_distribution.index[time_distribution.argmax()],
                        'review_count': max_cluster_size
                    })
        
        return clusters
        
        
    def calculate_centrality_metrics(self, node_id: str) -> Dict[str, float]:
        # Calculate various centrality measures
        metrics = {
            'degree_centrality': nx.degree_centrality(self.G)[node_id],
            'betweenness_centrality': nx.betweenness_centrality(self.G)[node_id]
        }
        
        # Try to calculate eigenvector centrality with increased max iterations
        try:
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(self.G, max_iter=1000)[node_id]
        except:
            metrics['eigenvector_centrality'] = None
            
        return metrics
    

    def detect_patterns(self, node_id: str) -> Dict[str, Any]:
        # Implement pattern detection
        patterns = {
            'suspicious_connections': self._find_suspicious_connections(node_id),
            'unusual_activity': self._detect_unusual_activity(node_id),
            'relationship_clusters': self._identify_clusters(node_id)
        }
        return patterns

    def _find_suspicious_connections(self, node_id: str) -> List[Dict[str, Any]]:
        # Example implementation for suspicious pattern detection
        suspicious_patterns = []
        node_type = self.G.nodes[node_id]['type']
        
        if node_type == 'Seller':
            # Check for unusual pricing patterns
            seller_offers = [n for n in self.G.neighbors(node_id) if self.G.nodes[n]['type'] == 'Offer']
            offer_prices = [float(self.df_dict['Offers'].loc[self.df_dict['Offers']['OfferID'] == o, 'Price'].values[0]) 
                          for o in seller_offers]
            
            if offer_prices:
                mean_price = np.mean(offer_prices)
                std_price = np.std(offer_prices)
                
                for offer, price in zip(seller_offers, offer_prices):
                    if abs(price - mean_price) > 2 * std_price:
                        suspicious_patterns.append({
                            'type': 'unusual_pricing',
                            'offer_id': offer,
                            'price': price,
                            'deviation': abs(price - mean_price) / std_price
                        })
        
        return suspicious_patterns

# Usage example:
def create_analysis_modules(G: nx.DiGraph, df_dict: Dict[str, pd.DataFrame]):
    viz = EnhancedVisualization(G)
    analytics = AdvancedAnalytics(G, df_dict)
    return viz, analytics

# Example usage:
"""
# In your main code:
viz, analytics = create_analysis_modules(G, df_dict)

# Create interactive visualization
fig = viz.create_interactive_plot(selected_id, depth)
fig = viz.add_minimap(fig)
st.plotly_chart(fig)

# Get analytics
metrics = analytics.calculate_centrality_metrics(selected_id)
patterns = analytics.detect_patterns(selected_id)

# Display results
st.write("Centrality Metrics:", metrics)
st.write("Detected Patterns:", patterns)
"""
