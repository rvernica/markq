# Reciprocal Rank Fusion

RRF fuses ranked lists from multiple retrievers into a single ranking. It
takes only the rank, not the per-list scores, which makes it robust to
score scale differences between BM25 and vector cosine.
