def calculate_hit_rate(recommanded_items:list, true_items:list):
    if not recommanded_items or not true_items:
        raise KeyError("missing items")
        return 0.0
    hits = 0
    total_cases = 0
    for top_k, next_picked in recommanded_items, true_items:        
        total_cases += 1
        if next_picked in top_k:
            hits += 1
    return hits / total_cases