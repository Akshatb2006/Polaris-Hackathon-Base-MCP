"""Exact product-name recall: python eval/name_recall.py [n]

Shoppers sometimes type a product's exact name. For a random sample of catalog names,
check that a product carrying that name appears in the top 10 (names are not unique, so
any product with the same name counts) and report which query words spelling changed.
"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search import Catalog  # noqa: E402

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    catalog = Catalog.load(str(Path(__file__).resolve().parent.parent / 'data/products.json'))
    by_id = {p['id']: p for p in catalog.products}
    rng = random.Random(3)
    sample = rng.sample(catalog.products, n)
    hits, empty, changed = 0, 0, Counter()
    misses = []
    for p in sample:
        out = catalog.search({'query': p['name']})
        names = {by_id[r['id']]['name'].casefold() for r in out['results']}
        hit = p['name'].casefold() in names
        hits += hit
        empty += not out['results']
        plan = catalog.engine.parser.parse(p['name'])
        changed.update(f'{a}->{b}' for a, b in plan.corrections.items())
        if not hit:
            misses.append((p['name'], plan.reason[:40], dict(plan.corrections)))
    print(f'name recall@10 {hits}/{n} = {hits / n:.1%}   empty responses {empty}')
    print('spelling changes applied to real names:', changed.most_common(15))
    for m in misses[:25]:
        print('  miss', m)
