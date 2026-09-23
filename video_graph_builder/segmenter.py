def scene_spans_from_boundaries(scene_boundaries, num_frames):
    last_idx = max(num_frames - 1, 0)
    cuts = sorted(set([0, last_idx] + list(scene_boundaries)))
    spans = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]
    return spans or [(0, last_idx)]


def merge_short_segments(spans, min_frames):
    if not spans:
        return spans

    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if (merged[-1][1] - merged[-1][0]) < min_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_frames:
        merged[-2][1] = merged[-1][1]
        merged.pop()

    return [tuple(span) for span in merged]


def build_segment_hierarchy(scene_spans, action_results, min_seconds=1.0, fps=1.0):
    """Split each scene span at its own action boundaries, returning
    [{"span": (start_idx, end_idx), "actions": [(start_idx, end_idx), ...]}]
    in sampled-frame-index space. Action spans never cross a scene boundary,
    and any action shorter than min_seconds is merged into its neighbor.

    Rule for which level represents a stretch of video: a scene gets action
    children only if it still has 2+ actions after merging; otherwise
    "actions" is empty and the scene node itself is the finest level there
    (so we never emit a child that duplicates its parent's span).
    """
    min_segment_frames = max(2, round(min_seconds * fps))

    hierarchy = []
    for (scene_start, scene_end), action_result in zip(scene_spans, action_results):
        cuts = {scene_start, scene_end}
        cuts.update(
            bp for bp in action_result.get('global_boundaries', [])
            if scene_start < bp < scene_end
        )
        sorted_cuts = sorted(cuts)
        action_spans = [
            (sorted_cuts[i], sorted_cuts[i + 1])
            for i in range(len(sorted_cuts) - 1)
            if sorted_cuts[i + 1] > sorted_cuts[i]
        ]
        action_spans = merge_short_segments(action_spans, min_segment_frames)
        hierarchy.append({
            "span": (scene_start, scene_end),
            "actions": action_spans if len(action_spans) > 1 else [],
        })

    return hierarchy
