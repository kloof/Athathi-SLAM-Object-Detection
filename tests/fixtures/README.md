# Test fixtures

## Synthetic fixtures (committed)

- `walls_closed_rectangle.py` — emits a list of 4 wall tuples forming a closed rectangle (Δ = 0). Used by Stage 8 no-op tests.
- `walls_open_lshape.py` — emits 6 wall tuples forming an L-shape with a 1 cm closure gap. Used by Stage 8 degeneracy tests.

## Non-committed fixtures (generated)

None yet. Future M0e upgrade: committing a 15-frame `mini_scan` MCAP (~30 MB) to enable end-to-end smoke tests without requiring the canonical scan path. Out of scope for this commit.
