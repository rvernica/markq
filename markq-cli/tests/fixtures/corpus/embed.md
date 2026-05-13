# Embedding worker

A single owner thread per model, fed by a bounded crossbeam channel, batches
chunks into 32 to 64 entry decode calls.
