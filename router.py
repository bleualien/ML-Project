"""
Take reasoning scores and produce final routing decisions and priority.
"""

def route_from_scores(scores: dict):
    """
    scores: { department: float (0..1) }
    returns routing dict:
      { departments: [list], priority: 'low'|'medium'|'high' }
    """
    # pick departments with score > threshold
    selected = [d for d, s in scores.items() if s > 0.35]
    # fallback: choose top 1 or 2
    if not selected:
        # pick top 2
        sorted_depts = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = [sorted_depts[0][0]]
        if len(sorted_depts) > 1 and sorted_depts[1][1] > 0.1:
            selected.append(sorted_depts[1][0])

    # determine priority by max score
    max_score = max([scores[d] for d in selected]) if selected else 0.0
    if max_score > 0.7:
        priority = 'high'
    elif max_score > 0.4:
        priority = 'medium'
    else:
        priority = 'low'

    return {"departments": selected, "priority": priority, "scores": scores}
