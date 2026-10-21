"""Local benchmark that mirrors the organizer runner: python eval/benchmark.py [URL]

1. validity: invalid calls must error; exact empty-query pages must equal an independent
   brute-force reference; repeated + prefix pagination probes must agree.
2. semantic: labelled shopping cases with machine-checkable expectations (garment kind,
   colour family, budget, audience, abstention, choice coverage, recommendation rules).
3. performance: 20 warm-up calls, then 50 calls at concurrency 1 and 5, repeated 3x;
   nearest-rank p95 and the official latency-credit formula.
"""
import asyncio
import json
import math
import random
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parent.parent
CATALOG = sorted(json.loads((ROOT / 'data/products.json').read_text()), key=lambda p: p['id'])
BY_ID = {p['id']: p for p in CATALOG}
CASES = json.loads((Path(__file__).parent / 'cases.json').read_text())
STRING_FILTERS = ('category', 'product_type', 'color', 'audience', 'section', 'pattern')


# ---------------------------------------------------------------- reference implementation
def ref_match(p, f):
    for k in STRING_FILTERS:
        if k in f and str(p.get(k) or '').strip().casefold() != f[k].strip().casefold():
            return False
    price, avg, cnt = p['price']['amount'], p['rating']['average'], p['rating']['count']
    return (price >= f.get('min_price', 0) and price <= f.get('max_price', math.inf)
            and ('min_rating' not in f or (avg is not None and avg >= f['min_rating']))
            and cnt >= f.get('min_review_count', 0))


def ref_page(args):
    f = args.get('filters', {})
    rows = [p for p in CATALOG if ref_match(p, f)]
    sort_by, desc = args.get('sort_by', 'relevance'), args.get('sort_order', 'desc') == 'desc'
    if sort_by == 'id':
        rows.sort(key=lambda p: p['id'], reverse=desc)
    elif sort_by == 'price':
        rows.sort(key=lambda p: (-p['price']['amount'] if desc else p['price']['amount'], p['id']))
    elif sort_by == 'review_count':
        rows.sort(key=lambda p: (-p['rating']['count'] if desc else p['rating']['count'], p['id']))
    elif sort_by == 'rating':
        rated = [p for p in rows if p['rating']['average'] is not None]
        rated.sort(key=lambda p: (-p['rating']['average'] if desc else p['rating']['average'], p['id']))
        rows = rated + [p for p in rows if p['rating']['average'] is None]
    o, l = args.get('offset', 0), args.get('limit', 10)
    return [p['id'] for p in rows[o:o + l]]


def random_filter_args(rng):
    args = {'filters': {}}
    for key in rng.sample(STRING_FILTERS, rng.randint(0, 2)):
        args['filters'][key] = rng.choice(CATALOG)[key].upper() if rng.random() < .3 else rng.choice(CATALOG)[key]
    if rng.random() < .5:
        lo = rng.choice([0, 199, 499, 999, 1999])
        args['filters']['min_price'] = lo
        args['filters']['max_price'] = lo + rng.choice([0, 500, 1500, 5000])
    if rng.random() < .3:
        args['filters']['min_rating'] = rng.choice([1, 3.5, 4, 4.5, 5])
    if rng.random() < .3:
        args['filters']['min_review_count'] = rng.choice([0, 1, 3, 6])
    args['sort_by'] = rng.choice(['relevance', 'id', 'price', 'rating', 'review_count'])
    args['sort_order'] = rng.choice(['asc', 'desc'])
    args['limit'] = rng.choice([1, 5, 10, 25, 100])
    args['offset'] = rng.choice([0, 0, 3, 50])
    if rng.random() < .3:
        args['query'] = '   '
    return args


INVALID = [{'limit': 0}, {'limit': 101}, {'offset': -1}, {'offset': 10001}, {'sort_by': 'name'},
           {'sort_order': 'up'}, {'query': None}, {'query': 'x' * 1001}, {'filters': {'color': ''}},
           {'filters': {'color': '  '}}, {'filters': {'min_price': 10, 'max_price': 5}},
           {'filters': {'min_rating': 0.5}}, {'filters': {'min_rating': 6}}, {'filters': {'size': 'M'}},
           {'filters': {'min_review_count': 1.5}}, {'filters': {'color': ['Black']}}, {'limit': '10'},
           {'include_recommendations': 'yes'}, {'unknown': 1}, {'filters': None}, {'limit': True}]


# ---------------------------------------------------------------- semantic checks
def product_ok(p, exp):
    text = f"{p['name']} {p['description'] or ''} {p['product_type']}".casefold()
    if 'types' in exp and p['product_type'] not in exp['types'] and not any(w in text for w in exp.get('words', [])):
        return False
    if 'colors' in exp and p['color'] not in exp['colors']:
        return False
    if 'max_price' in exp and p['price']['amount'] > exp['max_price']:
        return False
    if 'min_price' in exp and p['price']['amount'] < exp['min_price']:
        return False
    if 'audiences' in exp and p['audience'] not in exp['audiences']:
        return False
    if 'not_audiences' in exp and p['audience'] in exp['not_audiences']:
        return False
    return True


def ndcg_binary(flags, k=10):
    dcg = sum((1 if f else 0) / math.log2(i + 2) for i, f in enumerate(flags[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(k))
    return dcg / ideal


def grade_case(case, out):
    results = [BY_ID.get(r['id']) for r in out['results']]
    if any(r is None for r in results):
        return 0.0, 'unknown id'
    kind = case['kind']
    if kind == 'empty':
        return float(out == {'results': [], 'recommendations': []}), ''
    if kind in ('single', 'unusual'):
        flags = [product_ok(p, case['expect']) for p in results]
        return ndcg_binary(flags), f"{sum(flags[:10])}/10"
    if kind == 'mixed':
        covered = [any(product_ok(p, e) for p in results[:10]) for e in case['choices']]
        flags = [any(product_ok(p, e) for e in case['choices']) for p in results]
        return 0.5 * sum(covered) / len(covered) + 0.5 * ndcg_binary(flags), f"coverage {sum(covered)}/{len(covered)}"
    if kind == 'recs':
        primary = ndcg_binary([product_ok(p, case['expect']) for p in results])
        page = {r['id'] for r in out['results']}
        recs = out['recommendations']
        valid = [r for r in recs if r['for_product_id'] in page and r['id'] not in page]
        if len(valid) != len(recs) or len({r['id'] for r in recs}) != len(recs):
            return 0.0, 'invalid recommendation references'
        flags = [product_ok(BY_ID[r['id']], case['rec_expect']) for r in recs]
        rec_score = sum(f / math.log2(i + 2) for i, f in enumerate(flags)) / sum(1 / math.log2(i + 2) for i in range(5))
        return 0.5 * primary + 0.5 * rec_score, f"recs {sum(flags)}/{len(recs)}"
    raise ValueError(kind)


# ---------------------------------------------------------------- runner
async def call(session, args):
    t = time.perf_counter()
    res = await session.call_tool('search_products', args)
    return res, (time.perf_counter() - t) * 1000


async def main(url):
    rng = random.Random(7)
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print('== validity')
            inv = 0
            for args in INVALID:
                res, _ = await call(session, args)
                inv += bool(res.isError)
                if not res.isError:
                    print('   accepted invalid call:', args)
            exact, total = 0, 0
            for _ in range(80):
                args = random_filter_args(rng)
                res, _ = await call(session, args)
                got = [r['id'] for r in res.structuredContent['results']]
                total += 1
                if got == ref_page(args):
                    exact += 1
                else:
                    print('   exact mismatch:', json.dumps(args)[:200])
            stable = 0
            probes = [c['args'] for c in CASES if c['kind'] != 'empty'][:20]
            for args in probes:
                a, _ = await call(session, args | {'limit': 10})
                b, _ = await call(session, args | {'limit': 10})
                c, _ = await call(session, args | {'limit': 30})
                ra = [r['id'] for r in a.structuredContent['results']]
                stable += ra == [r['id'] for r in b.structuredContent['results']] == \
                    [r['id'] for r in c.structuredContent['results']][:len(ra)]
            print(f'   invalid rejected {inv}/{len(INVALID)} | exact pages {exact}/{total} | '
                  f'pagination stable {stable}/{len(probes)}')

            print('== semantic cases')
            by_kind, slow = {}, []
            for case in CASES:
                res, ms = await call(session, case['args'])
                if res.isError:
                    score, note = 0.0, 'ERROR ' + res.content[0].text[:80]
                else:
                    score, note = grade_case(case, res.structuredContent)
                by_kind.setdefault(case['kind'], []).append(score)
                flag = ' ' if score >= 0.7 else '!'
                print(f" {flag} {case['kind']:<7} {score:.2f} {ms:6.0f}ms  {case['args'].get('query', '')[:70]:<70} {note}")
                slow.append(ms)
            for kind, scores in by_kind.items():
                print(f'   {kind:<8} mean {sum(scores) / len(scores):.3f} over {len(scores)}')
            print(f'   functional latency max {max(slow):.0f} ms')

            print('== performance')
            pool = [random_filter_args(random.Random(1000 + i)) | {'limit': 10, 'offset': 0} for i in range(50)]
            # Expected pages are computed up front: the brute-force reference is CPU-heavy and,
            # run inside a timed concurrent workload, it would block the event loop and inflate
            # every in-flight request's latency (this produced a spurious 130 ms p95 earlier).
            expected = [ref_page(a) for a in pool]
            for args in pool[:20]:
                await call(session, args)
            timings, correct = [], 0
            for _ in range(3):
                for conc in (1, 5):
                    sem = asyncio.Semaphore(conc)

                    async def one(a, want):
                        async with sem:
                            res, ms = await call(session, a)
                        ok = [r['id'] for r in res.structuredContent['results']] == want
                        return ms if ok else 10_000, ok
                    out = await asyncio.gather(*(one(a, w) for a, w in zip(pool, expected)))
                    timings += [m for m, _ in out]
                    correct += sum(ok for _, ok in out)
            timings.sort()
            p50 = timings[len(timings) // 2]
            p95 = timings[math.ceil(0.95 * len(timings)) - 1]
            p99 = timings[math.ceil(0.99 * len(timings)) - 1]
            rate = correct / len(timings)
            credit = 5 * rate * max(0, min(1, (2000 - p95) / 1800))
            print(f'   p50 {p50:.1f} ms | p95 {p95:.1f} ms | p99 {p99:.1f} ms | correct {rate:.1%} | latency points {credit:.2f}/5')


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000/mcp'))
