"""Dense semantic channel: exact (brute-force) cosine search over bge-small embeddings.

A 105k x 384 float32 matrix-vector product takes a few milliseconds, so exact search
is both faster to operate and more correct than an approximate index: every product
that passes the hard filters is scored, so filtering never silently drops matches.
"""
import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

MODEL_NAME = os.getenv('EMBED_MODEL', 'BAAI/bge-small-en-v1.5')
INDEX_DIR = Path(os.getenv('INDEX_DIR', Path(__file__).resolve().parent.parent / 'index'))
MODEL_CACHE = os.getenv('FASTEMBED_CACHE', str(Path(__file__).resolve().parent.parent / '.models'))


def document_text(p):
    """One retrieval document per product. Synthetic reviews are deliberately excluded."""
    desc = (p.get('description') or '').strip()
    return (f"{p['name']}. {p['product_type']}. {desc} Colour: {p['color']}. "
            f"Pattern: {p['pattern']}. {p['category']}. For {p['audience']}, {p['section']}.")


def fingerprint(products):
    """Ties the stored embedding matrix to the exact catalog order and content."""
    h = hashlib.sha256()
    for p in products:
        h.update(p['id'].encode())
        h.update(document_text(p).encode())
    return h.hexdigest()[:16]


class DenseIndex:
    def __init__(self, matrix, model):
        self.matrix = matrix          # (n, d) float32, L2-normalised rows
        self.model = model
        self._lock = threading.Lock()
        self._cache = {}

    @classmethod
    def load(cls, products, allow_build=False):
        """Return a DenseIndex, or None when embeddings are unavailable (lexical-only mode)."""
        if os.getenv('DISABLE_DENSE') == '1':
            return None
        meta_path, vec_path = INDEX_DIR / 'meta.json', INDEX_DIR / 'embeddings.npy'
        fp = fingerprint(products)
        matrix = None
        if meta_path.exists() and vec_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get('fingerprint') == fp and meta.get('model') == MODEL_NAME:
                matrix = np.load(vec_path).astype(np.float32)
        if matrix is None and not allow_build:
            return None
        model = _model()
        if matrix is None:
            matrix = build(products, model)
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            np.save(vec_path, matrix.astype(np.float16))
            meta_path.write_text(json.dumps({'fingerprint': fp, 'model': MODEL_NAME,
                                             'dim': int(matrix.shape[1]), 'count': len(products),
                                             'normalised': True, 'metric': 'cosine'}, indent=2))
        return cls(matrix, model)

    def embed_query(self, text):
        if text in self._cache:
            return self._cache[text]
        with self._lock:
            vec = np.asarray(next(iter(self.model.query_embed(text))), dtype=np.float32)
        vec /= np.linalg.norm(vec) or 1.0
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[text] = vec
        return vec

    def embed_passages(self, texts):
        with self._lock:
            out = np.asarray(list(self.model.passage_embed(texts)), dtype=np.float32)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    def scores(self, text):
        return self.matrix @ self.embed_query(text)


def _model():
    from fastembed import TextEmbedding
    return TextEmbedding(MODEL_NAME, cache_dir=MODEL_CACHE)


def build(products, model, batch_size=256):
    texts = [document_text(p) for p in products]
    # Batch similar lengths together so little compute is spent on padding; rows are
    # restored to catalog order afterwards (row i always belongs to products[i]).
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    encoded = np.asarray(list(model.passage_embed([texts[i] for i in order], batch_size=batch_size)),
                         dtype=np.float32)
    vecs = np.empty_like(encoded)
    vecs[order] = encoded
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
