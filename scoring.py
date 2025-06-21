# scoring.py

def score_entity(entity_id, visited_nodes, graph):
    """
    Compute a risk score and abuse tags for the given entity using visited subgraph.

    Args:
        entity_id (str): ID of the entity being evaluated.
        visited_nodes (dict): Nodes visited from fan-out traversal.
        graph (networkx.DiGraph): Full graph.

    Returns:
        dict: {'risk_score': int, 'abuse_tags': list[str], 'reason': str}
    """
    node_type = graph.nodes[entity_id].get('type', 'Unknown')
    score = 0
    tags = []
    reasons = []

    # R1: Too many reviews connected
    reviews = [n for n in visited_nodes if graph.nodes[n].get('type') == 'Review']
    if len(reviews) > 20:
        score += 30
        tags.append("review_ring")
        reasons.append(f"{len(reviews)} reviews found in subgraph")

    # R2: Many orders but no reviews
    orders = [n for n in visited_nodes if graph.nodes[n].get('type') == 'Order']
    if len(orders) > 10 and len(reviews) == 0:
        score += 20
        tags.append("silent_orders")
        reasons.append(f"{len(orders)} orders found but no reviews")

    # R3: Self-review pattern for Sellers
    if node_type == 'Seller':
        offers = {n for n in visited_nodes if graph.nodes[n].get('type') == 'Offer'}
        customers = {n for n in visited_nodes if graph.nodes[n].get('type') == 'Customer'}
        for review in reviews:
            preds = list(graph.predecessors(review))
            succs = list(graph.successors(review))
            for customer in preds:
                if customer in customers:
                    for product in succs:
                        if graph.nodes[product].get('type') == 'ChildProduct':
                            related_offers = [o for o in graph.predecessors(product) if graph.nodes[o].get('type') == 'Offer']
                            for offer in related_offers:
                                seller_parents = [s for s in graph.predecessors(offer) if graph.nodes[s].get('type') == 'Seller']
                                if entity_id in seller_parents:
                                    score += 50
                                    tags.append("self_review")
                                    reasons.append(f"Customer {customer} reviewed product {product} offered by same seller")
                                    break

    return {
        'risk_score': score,
        'abuse_tags': list(set(tags)),
        'reason': "; ".join(reasons) if reasons else "No risky pattern detected"
    }
