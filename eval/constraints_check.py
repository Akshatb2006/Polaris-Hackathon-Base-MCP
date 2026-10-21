"""Hard-constraint property check on the full catalog: python eval/constraints_check.py

For each request, every primary result must satisfy the explicit filters AND the stated
query constraints; every recommendation must satisfy its own stated constraints, reference a
product on the page, and not repeat a primary result. Semantic similarity must never let a
product through that violates any of these.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search import Catalog  # noqa: E402

BLUE = {'Blue', 'Dark Blue', 'Light Blue', 'Other Blue'}
RED = {'Red', 'Dark Red', 'Light Red', 'Other Red'}
MEN = {'Menswear'}
CASES = [
    # (arguments, primary predicate, recommendation predicate or None)
    ({'query': 'red dress under 1500', 'limit': 50}, lambda p: p['color'] in RED and p['price']['amount'] <= 1500, None),
    ({'query': 'red dress', 'filters': {'color': 'Black'}, 'limit': 50}, lambda p: p['color'] == 'Black', None),
    ({'query': 'jeans', 'filters': {'product_type': 'Dress'}, 'limit': 30}, lambda p: p['product_type'] == 'Dress', None),
    ({'query': 'dress under 3000', 'filters': {'max_price': 1000}, 'limit': 50}, lambda p: p['price']['amount'] <= 1000, None),
    ({'query': 'dress over 3000', 'filters': {'max_price': 1000}, 'limit': 50}, lambda p: p['price']['amount'] <= 1000, None),
    ({'query': 'blue jeans for men between 1000 and 2500', 'limit': 50},
     lambda p: p['color'] in BLUE and 1000 <= p['price']['amount'] <= 2500 and
     (p['audience'] in MEN or p['section'] == 'Men H&M Sport'), None),
    ({'query': 'sneakers not white', 'limit': 50}, lambda p: p['color'] not in {'White', 'Off White'}, None),
    ({'query': 'tops rated 4 stars and above', 'limit': 50},
     lambda p: p['rating']['average'] is not None and p['rating']['average'] >= 4, None),
    ({'query': 'cheap hoodie', 'filters': {'min_rating': 4.5, 'min_review_count': 3}, 'limit': 50},
     lambda p: p['rating']['average'] is not None and p['rating']['average'] >= 4.5 and p['rating']['count'] >= 3, None),
    ({'query': 'Black jeans and white T-shirts under INR 2000 each', 'filters': {'audience': 'Ladieswear'}, 'limit': 40},
     lambda p: p['audience'] == 'Ladieswear' and p['price']['amount'] <= 2000 and p['color'] in {'Black', 'White', 'Off White'}, None),
    ({'query': 'A dress for dinner under INR 2000 with a matching bag under INR 1000', 'include_recommendations': True},
     lambda p: p['price']['amount'] <= 2000,
     lambda r: r['price']['amount'] <= 1000 and r['product_type'] in
     {'Bag', 'Weekend/Gym bag', 'Backpack', 'Cross-body bag', 'Tote bag', 'Shoulder bag', 'Bumbag'}),
    ({'query': 'shirt for men with a matching black tie', 'include_recommendations': True},
     lambda p: p['audience'] in MEN or p['section'] == 'Men H&M Sport',
     lambda r: r['color'] == 'Black' and r['product_type'] == 'Tie'),
    ({'query': 'kids jacket', 'filters': {'max_price': 1500}, 'include_recommendations': True},
     lambda p: p['price']['amount'] <= 1500, lambda r: True),
    ({'query': 'swimsuit', 'sort_by': 'price', 'sort_order': 'asc', 'limit': 100}, lambda p: True, None),
]


def main():
    catalog = Catalog.load(str(Path(__file__).resolve().parent.parent / 'data/products.json'))
    by_id = {p['id']: p for p in catalog.products}
    failures = 0
    for args, ok, rec_ok in CASES:
        out = catalog.search(args)
        res = [by_id[r['id']] for r in out['results']]
        bad = [p['id'] for p in res if not ok(p)]
        page = {r['id'] for r in out['results']}
        rec_bad = []
        for r in out['recommendations']:
            if r['id'] in page or r['for_product_id'] not in page or not (rec_ok or (lambda _: False))(by_id[r['id']]):
                rec_bad.append(r['id'])
        if len({r['id'] for r in out['recommendations']}) != len(out['recommendations']):
            rec_bad.append('duplicate')
        if args.get('sort_by') == 'price':
            prices = [p['price']['amount'] for p in res]
            if prices != sorted(prices):
                bad.append('price order')
        status = 'PASS' if not bad and not rec_bad and res else 'FAIL'
        failures += status == 'FAIL'
        print(f"{status}  {len(res):>3} results  {len(out['recommendations'])} recs  {args.get('query')!r} "
              f"{args.get('filters', '')} {('bad=' + str(bad[:3])) if bad else ''} {('rec_bad=' + str(rec_bad[:3])) if rec_bad else ''}")
    print(f'{len(CASES) - failures}/{len(CASES)} passed')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
