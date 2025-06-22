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
        mini_fig = go.Figure()
        
        # Add simplified nodes
        mini_fig.add_trace(go.Scatter(
            x=[mini_pos[node][0] for node in self.G.nodes()],
            y=[mini_pos[node][1] for node in self.G.nodes()],
            mode='markers',
            marker=dict(size=2)
        ))
        
        return mini_fig
    
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

    def _detect_unusual_activity(self, node_id: str) -> List[Dict[str, Any]]:
        unusual_activities = []
        node_type = self.G.nodes[node_id]['type']
        # Add your unusual activity detection logic here
        return unusual_activities
    
    def _identify_clusters(self, node_id: str) -> List[Dict[str, Any]]:
        clusters = []
        # Add your clustering logic here
        return clusters
        
    def calculate_centrality_metrics(self, node_id: str) -> Dict[str, float]:
        # Calculate various centrality measures
        metrics = {
            'degree_centrality': nx.degree_centrality(self.G)[node_id],
            'betweenness_centrality': nx.betweenness_centrality(self.G)[node_id],
            'eigenvector_centrality': nx.eigenvector_centrality(self.G)[node_id]
        }
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
