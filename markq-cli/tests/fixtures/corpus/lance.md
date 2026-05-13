# LanceDB backend

LanceDB stores chunks as a single Lance dataset with a collection column.
Cross-collection queries are filter pushdown rather than union scans.

## FTS

LanceDB full text search uses a Tantivy-derived BM25 implementation.
