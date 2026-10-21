"""Rule-based graded relevance judge: python eval/judge.py [--split dev|heldout|all] [--mode hybrid|...]

Approximates the organizer's rubric (0 irrelevant, 1 weak, 2 useful but less direct, 3 directly
relevant) from catalog facts only, and scores nDCG@10 against a fixed ideal of ten grade-3 slots
exactly as the brief defines it. Labels live in eval/judge_cases.json and were written from the
shopper's point of view, independently of the engine's taxonomy. It is a proxy for comparing
versions of this engine, not a prediction of the organizer's Bedrock scores.

Per product:
  0  wrong garment kind, any hard requirement violated (colour, budget, audience), or an
     "avoid" phrase present
  1  right kind via a secondary type only ("also"), or no preferred attribute matched
  2  right kind and about half the preferred attributes present
  3  right kind and (nearly) all preferred attributes present
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# ENGINE_ROOT lets the same cases and grading score another checkout (e.g. the deployed commit).
ENGINE_ROOT = Path(__import__('os').getenv('ENGINE_ROOT', ROOT))
sys.path.insert(0, str(ENGINE_ROOT))

CASES = json.loads((Path(__file__).parent / 'judge_cases.json').read_text())


def text_of(p):
    return f"{p['name']} {p.get('description') or ''} {p['product_type']} {p['section']}".casefold()


def kind_level(p, spec):
    """2 = primary kind, 1 = acceptable secondary kind, 0 = wrong kind."""
    text = text_of(p)
    head = (p.get('description') or '').casefold()[:60]   # the garment noun opens the description
    if p['product_type'] in spec.get('types', []) or any(w in head for w in spec.get('words', [])):
        return 2
    if p['product_type'] in spec.get('also', []) or any(w in text for w in spec.get('also_words', [])):
        return 1
    return 0 if ('types' in spec or 'words' in spec) else 2


def violates(p, spec):
    """Hard requirements stated by the shopper (colour family, budget, audience)."""
    if 'colors' in spec and p['color'] not in spec['colors']:
        return True
    if 'max_price' in spec and p['price']['amount'] > spec['max_price']:
        return True
    if 'min_price' in spec and p['price']['amount'] < spec['min_price']:
        return True
    return 'audiences' in spec and p['audience'] not in spec['audiences'] \
        and p['section'] not in spec.get('sections', [])


def grade(p, spec):
    text = text_of(p)
    if 'name' in spec:   # exact product-name lookup: the named product (any variant) is the answer
        if p['name'].casefold() == spec['name'].casefold():
            return 3
        return 1 if p['product_type'] == spec.get('type') else 0
    if violates(p, spec):
        return 0
    if any(w in text for w in spec.get('avoid', [])):
        return 0
    level = kind_level(p, spec)
    if level == 0:
        return 0
    prefer = spec.get('prefer', [])
    if prefer:
        # each preference is a word or a list of alternatives
        hits = sum(any(w in text for w in ([x] if isinstance(x, str) else x)) for x in prefer)
        frac = hits / len(prefer)
        g = 1 if frac == 0 else 2 if frac < 0.75 else 3
    else:
        g = 3
    return min(g, 2) if level == 1 else g


def ndcg(grades, k=10):
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))
    ideal = sum((2 ** 3 - 1) / math.log2(i + 2) for i in range(k))
    return dcg / ideal


def dcg_at(grades, k):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def score_case(case, out, by_id):
    """Returns (score in [0, 1], details). Ideal = 10 primary (or 5 recommendation) grade-3 slots."""
    res = [by_id[r['id']] for r in out['results']]
    if case['kind'] == 'empty':
        return float(not out['results'] and not out['recommendations']), {}
    if case['kind'] == 'recs':
        primary = [grade(p, case['expect']) for p in res]
        page = {r['id'] for r in out['results']}
        recs = out['recommendations']
        bad_ref = any(r['for_product_id'] not in page or r['id'] in page for r in recs) \
            or len({r['id'] for r in recs}) != len(recs)
        rec_grades = [grade(by_id[r['id']], case['rec_expect']) for r in recs]
        rec_viol = sum(violates(by_id[r['id']], case['rec_expect']) for r in recs)
        rec_score = 0.0 if bad_ref else dcg_at(rec_grades, 5) / sum(7 / math.log2(i + 2) for i in range(5))
        return 0.5 * ndcg(primary) + 0.5 * rec_score, {'grades': primary[:10], 'rec_grades': rec_grades,
                                                       'violations': rec_viol + sum(violates(p, case['expect']) for p in res[:10])}
    if case['kind'] == 'mixed':
        grades = [max(grade(p, c) for c in case['choices']) for p in res]
        covered = [any(grade(p, c) >= 2 for p in res[:10]) for c in case['choices']]
        cov = sum(covered) / len(covered)
        return 0.5 * cov + 0.5 * ndcg(grades), {'coverage': cov, 'grades': grades[:10], 'violations': 0}
    grades = [grade(p, case['expect']) for p in res]
    return ndcg(grades), {'grades': grades[:10], 'violations': sum(violates(p, case['expect']) for p in res[:10])}


def run(engine, cases, by_id, verbose=False):
    by_kind, details, times, violations = {}, [], [], 0
    for case in cases:
        args = {'query': case['query']} | case.get('args', {})
        t = time.perf_counter()
        out = engine.search(args)
        times.append((time.perf_counter() - t) * 1000)
        s, info = score_case(case, out, by_id)
        violations += info.get('violations', 0)
        by_kind.setdefault(case['kind'], []).append(s)
        details.append((s, case, info))
    if verbose:
        for s, case, info in sorted(details, key=lambda d: d[0]):
            if s < 0.95:
                print(f"  {s:.2f} [{case['split']}/{case['kind']}] {case['query'][:60]:<60} {info.get('grades', '')}"
                      f"{' recs ' + str(info['rec_grades']) if 'rec_grades' in info else ''}")
    summary = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_kind.items())}
    summary['all'] = round(sum(sum(v) for v in by_kind.values()) / sum(len(v) for v in by_kind.values()), 4)
    summary['hard_violations_top10'] = violations
    summary['cases'] = len(cases)
    times.sort()
    summary['p95_ms'] = round(times[math.ceil(0.95 * len(times)) - 1], 1)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='all')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--modes', default='hybrid', help='comma list of hybrid,lexical,dense')
    a = ap.parse_args()
    import engine.engine as eng
    from search import Catalog
    catalog = Catalog.load(str(ROOT / 'data/products.json'))
    print('engine from', Path(sys.modules['search'].__file__).parent)
    by_id = {p['id']: p for p in catalog.products}
    cases = [c for c in CASES if a.split == 'all' or c['split'] == a.split]
    calls = {'embed': 0, 'bm25': 0}
    embed, bm25 = catalog.engine.dense.embed_query, catalog.engine.index.bm25

    def count(name, fn):
        def wrapped(*args):
            calls[name] += 1
            return fn(*args)
        return wrapped
    catalog.engine.dense.embed_query = count('embed', embed)
    catalog.engine.index.bm25 = count('bm25', bm25)
    for mode in a.modes.split(','):
        eng.SEARCH_MODE = mode
        catalog.engine._cache.clear()
        catalog.engine.dense._cache.clear()
        calls.update(embed=0, bm25=0)
        summary = run(catalog.engine, cases, by_id, a.verbose)
        print(mode, json.dumps(summary | {'embedding_calls': calls['embed'], 'bm25_calls': calls['bm25']}))
