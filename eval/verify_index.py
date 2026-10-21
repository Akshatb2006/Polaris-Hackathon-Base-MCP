"""Index integrity: python eval/verify_index.py

Checks the stored dense matrix against the catalog: row count, dimension, finite values,
unit norms, fingerprint, and that re-encoding a random sample of products reproduces the
exact row stored for that product ID (so row i really is catalog product i).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.dense import INDEX_DIR, MODEL_NAME, _model, document_text, fingerprint  # noqa: E402

products = sorted(json.loads((INDEX_DIR.parent / 'data/products.json').read_text()), key=lambda p: p['id'])
meta = json.loads((INDEX_DIR / 'meta.json').read_text())
matrix = np.load(INDEX_DIR / 'embeddings.npy').astype(np.float32)
norms = np.linalg.norm(matrix, axis=1)
checks = {
    'rows == products': matrix.shape[0] == len(products) == meta['count'],
    'dim == 384': matrix.shape[1] == meta['dim'] == 384,
    'model matches': meta['model'] == MODEL_NAME,
    'fingerprint matches catalog': meta['fingerprint'] == fingerprint(products),
    'all values finite': bool(np.isfinite(matrix).all()),
    'unit norms (|n-1| < 1e-2)': bool(np.abs(norms - 1).max() < 1e-2),
    'no duplicate zero rows': bool((norms > 0.5).all()),
}
rng = random.Random(42)
sample = sorted(rng.sample(range(len(products)), 300))
model = _model()
fresh = np.asarray(list(model.passage_embed([document_text(products[i]) for i in sample])), dtype=np.float32)
fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
own = np.sum(fresh * matrix[sample], axis=1)
sims = fresh @ matrix.T
checks['re-encoded rows match stored rows (cos > 0.999)'] = bool(own.min() > 0.999)
# Catalog variants can embed identically (bge is uncased); the product's own row must tie for first.
checks["product's own row is its nearest neighbour (ties within 1e-4)"] = bool(
    (sims[np.arange(len(sample)), sample] >= sims.max(axis=1) - 1e-4).all())
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print(f'min self-cosine {own.min():.6f}  norm range [{norms.min():.4f}, {norms.max():.4f}]  shape {matrix.shape}')
sys.exit(0 if all(checks.values()) else 1)
