"""Offline index build: python build_index.py [catalog_path]

Encodes every product once with the dense model and stores a float16 matrix plus
build metadata (model, dimension, catalog fingerprint) in index/. The server loads it
at startup and refuses a stale matrix whose fingerprint does not match the catalog.
"""
import json
import sys
import time
from pathlib import Path

from engine.dense import DenseIndex


def main(path):
    products = sorted(json.loads(Path(path).read_text(encoding='utf-8')), key=lambda p: p['id'])
    start = time.time()
    index = DenseIndex.load(products, allow_build=True)
    print(f'dense index ready: {index.matrix.shape} in {time.time() - start:.1f}s')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / 'data/products.json'))
