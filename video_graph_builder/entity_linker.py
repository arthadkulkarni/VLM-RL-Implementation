"""Entity linking in two stages, both decided by the VLM's pairwise
same-entity verdicts:

1. Local: mentions in segments at most window_sec apart are compared
   directly (candidate_mention_pairs). Continuity makes these reliable, and
   the result is a set of tracklets -- runs of the same entity.
2. Global re-identification: tracklets are compared with each other across
   the whole video (candidate_tracklet_pairs), so an entity that leaves and
   comes back minutes later is still recognised. There is no shortlist:
   every same-type tracklet pair is a candidate, up to a per-video cap.

cluster_mentions merges "same" verdicts under two guards (see its
docstring); canonical_entities turns the final clusters into graph
entities.
"""


def _norm(value):
    return str(value or "").strip().lower()


def _gap_sec(seg_a, seg_b):
    """Seconds between two segments; <= 0 when they overlap (e.g. a scene and
    one of its own action children)."""
    return max(seg_a["start_sec"], seg_b["start_sec"]) - min(seg_a["end_sec"], seg_b["end_sec"])


def candidate_mention_pairs(mentions, segments_by_id, window_sec):
    """Stage 1: mention index pairs from different segments, same entity type,
    and segments at most window_sec apart. Returned closest-first, the order
    cluster_mentions expects its verdicts in.
    """
    pairs = []
    for i, a in enumerate(mentions):
        for j in range(i + 1, len(mentions)):
            b = mentions[j]
            if a["segment_id"] == b["segment_id"] or _norm(a.get("type")) != _norm(b.get("type")):
                continue
            gap = _gap_sec(segments_by_id[a["segment_id"]], segments_by_id[b["segment_id"]])
            if gap <= window_sec:
                pairs.append((gap, i, j))
    return [(i, j) for _, i, j in sorted(pairs)]


def candidate_tracklet_pairs(mentions, labels, segments_by_id, known_verdicts, max_pairs):
    """Stage 2: one representative mention pair per pair of stage-1 tracklets
    (clusters in `labels`) that have the same type, share no segment, and
    were not already compared in stage 1. The representatives are the two
    mentions closest in time, where appearance is most likely to match.

    Every candidate is kept up to max_pairs. Past the cap, pairs of the most
    prominent tracklets (by the smaller one's segment count) are kept first,
    since those entities anchor the most Planner candidates. Returns
    (pairs, total_candidates) so callers can report how much the cap dropped.
    """
    tracklets = {}
    for idx, label in enumerate(labels):
        tracklets.setdefault(label, []).append(idx)

    compared = {frozenset((labels[i], labels[j])) for i, j, _ in known_verdicts}
    tracklet_segments = {
        label: {mentions[i]["segment_id"] for i in members} for label, members in tracklets.items()
    }

    candidates = []
    ordered = list(tracklets)
    for x, label_a in enumerate(ordered):
        for label_b in ordered[x + 1:]:
            members_a, members_b = tracklets[label_a], tracklets[label_b]
            if _norm(mentions[members_a[0]].get("type")) != _norm(mentions[members_b[0]].get("type")):
                continue
            if frozenset((label_a, label_b)) in compared:
                continue
            if tracklet_segments[label_a] & tracklet_segments[label_b]:
                continue
            gap, i, j = min(
                (_gap_sec(segments_by_id[mentions[i]["segment_id"]], segments_by_id[mentions[j]["segment_id"]]), i, j)
                for i in members_a
                for j in members_b
            )
            prominence = min(len(tracklet_segments[label_a]), len(tracklet_segments[label_b]))
            candidates.append((-prominence, gap, min(i, j), max(i, j)))

    candidates.sort()
    return [(i, j) for _, _, i, j in candidates[:max_pairs]], len(candidates)


def cluster_mentions(mentions, verdicts):
    """Cluster labels (one per mention) from pairwise verdicts
    [(i, j, same)], merging "same" pairs in the order given. Two clusters are
    only merged if no pair between them was judged different, and never if
    they hold two mentions from the same segment (distinct entities by
    construction). Without these guards one wrong "same" verdict chains two
    different entities -- and everything linked to them -- into one.
    """
    parent = list(range(len(mentions)))
    members = [[i] for i in range(len(mentions))]
    cluster_segments = [{m["segment_id"]} for m in mentions]
    different = [set() for _ in mentions]
    for i, j, same in verdicts:
        if not same:
            different[i].add(j)
            different[j].add(i)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, same in verdicts:
        if not same:
            continue
        root_i, root_j = find(i), find(j)
        if root_i == root_j or cluster_segments[root_i] & cluster_segments[root_j]:
            continue
        if any(find(other) == root_j for m in members[root_i] for other in different[m]):
            continue
        parent[root_j] = root_i
        members[root_i].extend(members[root_j])
        cluster_segments[root_i] |= cluster_segments[root_j]

    return [find(i) for i in range(len(mentions))]


def canonical_entities(mentions, labels):
    """One graph entity per cluster. Each mention dict needs keys: name, type,
    description, segment_id."""
    clusters = {}
    for mention, label in zip(mentions, labels):
        clusters.setdefault(label, []).append(mention)

    def _most_common(values):
        counts = {}
        for value in values:
            if value:
                counts[value] = counts.get(value, 0) + 1
        return max(counts, key=counts.get) if counts else ""

    entities = []
    for idx, cluster_mentions in enumerate(clusters.values()):
        descriptions = [m.get("description", "") for m in cluster_mentions if m.get("description")]
        segment_ids = sorted({m.get("segment_id") for m in cluster_mentions if m.get("segment_id")})

        entities.append({
            "entity_id": f"ent_{idx}",
            "name": _most_common(m.get("name", "") for m in cluster_mentions),
            "type": _most_common(m.get("type", "") for m in cluster_mentions),
            "description": max(descriptions, key=len) if descriptions else "",
            "segment_ids": segment_ids,
        })

    return entities
