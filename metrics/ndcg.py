import math

def calculate_cdng(recommended_lists, picked_items):
    """
    Calculates an average Discounted Cumulative Gain (DCG) across multiple users,
    assuming exactly one 'true' item (picked) per user.

    Args:
        recommended_lists (list of list): Each element is a list of recommended items for a user.
        picked_items (list): A list of the single 'true' item each user actually picked,
                             in the same order as recommended_lists.

    Returns:
        float: The average DCG across all users. If recommended_lists or picked_items is empty,
               returns 0.0.
    
    Example:
        recommended_lists = [
            ["item_2", "item_9", "item_1", "item_5", "item_4"],  # user A's recommendations
            ["item_3", "item_6", "item_7", "item_1", "item_4"],  # user B's recommendations
            ...
        ]
        picked_items = [
            "item_9",  # user A's actually chosen item
            "item_6",  # user B's chosen item
            ...
        ]
        avg_cdng = calculate_cdng(recommended_lists, picked_items)
    """
    if not recommended_lists or not picked_items:
        return 0.0
    
    total_cases = 0
    sum_dcg = 0.0
    
    for recs, picked in zip(recommended_lists, picked_items):
        total_cases += 1
        if picked in recs:
            rank = recs.index(picked)  # 0-based
            sum_dcg += 1.0 / math.log2(rank + 2)
    
    if total_cases == 0:
        return 0.0
    
    return sum_dcg / total_cases
