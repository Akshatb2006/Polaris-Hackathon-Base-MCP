"""Side-by-side top-k per retrieval mode: python eval/compare_modes.py ["query" ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine.engine as eng  # noqa: E402
from search import Catalog  # noqa: E402

QUERIES = ['flowy white summer dress', 'something sparkly for new years eve party', 'cozy oversized knit sweater',
           'boho festival outfit', 'minimalist black bag for work', 'vintage washed denim jacket',
           'breathable linen trousers', 'chunky platform sneakers', 'I am always cold at the office',
           'wedding guest outfit', 'elegant evening dress', 'beach cover up', 'sporty look for men',
           'what should I wear to the beach', 'smth cute 2 wear on a 1st date']

if __name__ == '__main__':
    catalog = Catalog.load(str(Path(__file__).resolve().parent.parent / 'data/products.json'))
    for q in sys.argv[1:] or QUERIES:
        print(f'\n=== {q}')
        for mode in ('lexical', 'dense', 'hybrid'):
            eng.SEARCH_MODE = mode
            out = catalog.engine.run({'query': q, 'limit': 6}, explain=True)
            print(f'  [{mode}]')
            for p in out['products']:
                head = (p['description'] or '')[:55]
                print(f"     {p['product_type'][:14]:<14} {p['color'][:12]:<12} {p['price']:>5} {p['name'][:26]:<26} | {head}")
