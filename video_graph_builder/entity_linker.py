from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


def link_entities(mentions, distance_threshold=0.35):
    """Cluster per-segment entity mentions into canonical identities using
    TF-IDF text similarity over "type name description", the same
    AgglomerativeClustering(metric="precomputed") pattern already used for
    skill clustering in train_examples/reward_function/cot_val.py. Each
    mention dict needs keys: name, type, description, segment_id.
    """
    if not mentions:
        return []

    texts = [
        f"{m.get('type', '')} {m.get('name', '')} {m.get('description', '')}".strip()
        for m in mentions
    ]

    if len(mentions) == 1:
        labels = [0]
    else:
        try:
            vectors = TfidfVectorizer().fit_transform(texts)
            distances = cosine_distances(vectors)
        except ValueError:
            # empty vocabulary (e.g. every mention text is blank) -- keep mentions separate
            labels = list(range(len(mentions)))
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="precomputed",
                linkage="average",
            )
            labels = clustering.fit_predict(distances)

    clusters = {}
    for mention, label in zip(mentions, labels):
        clusters.setdefault(label, []).append(mention)

    def _most_common(values):
        counts = {}
        for value in values:
            if value:
                counts[value] = counts.get(value, 0) + 1
        return max(counts, key=counts.get) if counts else ""

    canonical_entities = []
    for idx, cluster_mentions in enumerate(clusters.values()):
        descriptions = [m.get("description", "") for m in cluster_mentions if m.get("description")]
        segment_ids = sorted({m.get("segment_id") for m in cluster_mentions if m.get("segment_id")})

        canonical_entities.append({
            "entity_id": f"ent_{idx}",
            "name": _most_common(m.get("name", "") for m in cluster_mentions),
            "type": _most_common(m.get("type", "") for m in cluster_mentions),
            "description": max(descriptions, key=len) if descriptions else "",
            "segment_ids": segment_ids,
        })

    return canonical_entities
