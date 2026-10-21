"""Search engine: query plan -> hard-constraint masks -> hybrid scoring -> pages -> outfit recommendations."""
import json
import os
import threading
from collections import OrderedDict

import numpy as np

from .dense import DenseIndex
from .index import CatalogIndex
from .nlu import Intent, QueryParser
from .taxonomy import COMPLEMENTS, CONCEPTS, HUE, NEUTRALS, ROLE_COMPLEMENTS

# Score blend (all channels normalised to [0, 1] over the candidate set).
W_DENSE, W_LEX, W_CONCEPT, W_PATTERN, W_OCCASION, W_QUALITY = 0.42, 0.26, 0.22, 0.14, 0.12, 0.035
# Without a named garment the request is about intent ("lazy Sunday", "cold at the office"),
# where literal word matches mislead; measured on eval/cases.json the dense channel wins there.
W_DENSE_OPEN, W_LEX_OPEN = 0.55, 0.12
FIELD_SORT_KEEP = 0.72       # for field sorts keep candidates scoring >= this fraction of the best
FIELD_SORT_CAP = 400
MAX_CANDIDATES = 10_100      # offset (<= 10000) + limit (<= 100)
CONCEPT_MIN = float(os.getenv('CONCEPT_MIN', '0.85'))   # type or description head; a name-only word is not proof
CHILD_PRIOR = 0.2            # unqualified requests are for adults; children's items rank after adult ones
W_EXACT_COLOR = 0.25         # 'grey' means Grey before Dark Grey or Light Grey
W_PRIMARY_TYPE = 0.15        # 'jacket' means product_type Jacket before a blazer described as a jacket
DEMOTE_PENALTY = 0.35         # products the request rules out by role (bridal gowns for a guest)
NAME_REPEAT_PENALTY = float(os.getenv('NAME_REPEAT_PENALTY', '0'))   # variants are full-credit slots to a relevance judge
# Ablation switch for evaluation: hybrid (default), lexical (no dense channel) or dense (no BM25F).
SEARCH_MODE = os.getenv('SEARCH_MODE', 'hybrid')
# Experimental, off by default: cross-encoder reordering of the top RERANK_TOP eligible candidates.
RERANK_MODEL = os.getenv('RERANK_MODEL', '')
RERANK_TOP, RERANK_WEIGHT = int(os.getenv('RERANK_TOP', '30')), float(os.getenv('RERANK_WEIGHT', '0.5'))


class SearchEngine:
    def __init__(self, products, dense='auto'):
        self.index = CatalogIndex(products)
        self.products = self.index.products
        self.parser = QueryParser(self.index)
        self.dense = DenseIndex.load(self.products) if dense == 'auto' else dense
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self.reranker = None
        if RERANK_MODEL:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self.reranker = TextCrossEncoder(RERANK_MODEL, cache_dir=os.getenv('RERANK_CACHE') or None)

    # ------------------------------------------------------------------ API
    def search(self, arguments):
        key = json.dumps(arguments, sort_keys=True)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return json.loads(self._cache[key])
        result = self.run(arguments)['response']
        with self._lock:
            self._cache[key] = json.dumps(result)
            if len(self._cache) > 2048:
                self._cache.popitem(last=False)
        return result

    def run(self, arguments, explain=False):
        query = arguments.get('query', '').strip()
        filters = arguments.get('filters', {})
        sort_by = arguments.get('sort_by', 'relevance')
        sort_order = arguments.get('sort_order', 'desc')
        offset, limit = arguments.get('offset', 0), arguments.get('limit', 10)
        want_recs = arguments.get('include_recommendations', False)
        trace = {}
        base = self.index.filter_mask(filters)
        if not query:
            # Exact catalog path: presorted permutation + boolean mask, no Python per-product loop.
            order = self.index.order(sort_by, sort_order)
            ranked = order[base[order]]
            page = ranked[offset:offset + limit]
            plan, intents_ranked = None, []
        else:
            plan = self.parser.parse(query)
            trace['plan'] = plan.describe()
            if not plan.shoppable:
                return self._response([], [], trace, explain)
            intents_ranked = [self._rank_intent(i, filters, base, trace if explain else None,
                                                plan.exact_name if not plan.mixed else ())
                              for i in plan.intents]
            ranked = self._merge(intents_ranked, sort_by, sort_order)
            page = ranked[offset:offset + limit]
        recs = []
        if want_recs and len(page):
            recs = self._recommend(page, plan, intents_ranked, filters)
        return self._response(page, recs, trace, explain)

    def _response(self, page, recs, trace, explain):
        response = {'results': [{'id': self.index.ids[i]} for i in page], 'recommendations': recs}
        out = {'response': response}
        if explain:
            out['trace'] = trace
            out['products'] = [self._card(i) for i in page]
            out['rec_products'] = [self._card(self.index.pos[r['id']]) for r in recs]
        return out

    def _card(self, i):
        p = self.products[i]
        return {k: p.get(k) for k in ('id', 'name', 'product_type', 'category', 'audience', 'section', 'color',
                                      'pattern', 'image_url', 'description')} | {
            'price': p['price']['amount'], 'rating': p['rating']['average'], 'reviews': p['rating']['count']}

    # ------------------------------------------------------------------ hard constraints
    def _intent_mask(self, intent, filters, base):
        """Query-derived hard constraints; explicit filters win over conflicting query wording."""
        idx = self.index
        mask = base.copy()
        notes = []
        lo = intent.min_price if 'min_price' not in filters else None
        hi = intent.max_price if 'max_price' not in filters else None
        if lo is not None and 'max_price' in filters and lo > filters['max_price']:
            lo = None
        if hi is not None and 'min_price' in filters and hi < filters['min_price']:
            hi = None
        if lo is not None:
            mask &= idx.price >= lo
        if hi is not None:
            mask &= idx.price <= hi
        if intent.min_rating is not None and 'min_rating' not in filters:
            with np.errstate(invalid='ignore'):
                mask &= ~np.isnan(idx.rating) & (idx.rating >= intent.min_rating)
        if intent.who and not ({'audience', 'section'} & filters.keys()):
            mask &= idx.who[intent.who]
        if intent.colors and 'color' not in filters:
            mask &= idx.color_mask(intent.colors)
        if intent.exclude_colors:
            mask &= ~idx.color_mask(intent.exclude_colors)
        if intent.brand:
            mask &= idx.brand_mask(intent.brand)
        for concept in intent.exclude_concepts:
            mask &= idx.concept_strength[concept] < 0.85
        concept_strength = None
        if intent.concepts:
            concept_strength = np.max([idx.concept_strength[c] for c in intent.concepts], axis=0)
            with_concept = mask & (concept_strength >= CONCEPT_MIN)
            conflicts = {'product_type', 'category'} & filters.keys()
            if with_concept.any() or not conflicts:
                mask = with_concept
            else:
                notes.append('explicit product filter overrides query garment')
                concept_strength = None
        return mask, concept_strength, notes

    # ------------------------------------------------------------------ scoring
    def _score(self, intent, mask, concept_strength):
        idx = self.index
        cand = np.flatnonzero(mask)
        if not len(cand):
            return cand, np.zeros(0, dtype=np.float32), {}
        parts = {}
        terms = intent.lexical_terms() if SEARCH_MODE != 'dense' or self.dense is None else []
        lex = idx.bm25(terms)[cand] if terms else np.zeros(len(cand), dtype=np.float32)
        parts['lexical'] = lex / lex.max() if lex.max() > 0 else lex
        if self.dense is not None and intent.semantic_text and SEARCH_MODE != 'lexical':
            dense = self.dense.scores(intent.semantic_text)[cand]
            lo = np.percentile(dense, 50) if len(dense) > 20 else dense.min()
            hi = dense.max()
            parts['dense'] = np.clip((dense - lo) / (hi - lo), 0, 1) if hi > lo else np.ones_like(dense)
            parts['dense_raw'] = dense
        else:
            parts['dense'] = parts['lexical']
        parts['concept'] = concept_strength[cand] if concept_strength is not None else np.zeros(len(cand), np.float32)
        parts['pattern'] = (idx.pattern_mask(intent.patterns)[cand].astype(np.float32) if intent.patterns
                            else np.zeros(len(cand), np.float32))
        if intent.exclude_patterns:
            parts['pattern'] -= idx.pattern_mask(intent.exclude_patterns)[cand]
        if intent.occasion_concepts:
            occ = np.max([idx.concept_strength[c][cand] * intent.occasion_weights.get(c, 1.0)
                          for c in intent.occasion_concepts if c in idx.concept_strength], axis=0)
            parts['occasion'] = occ
        else:
            parts['occasion'] = np.zeros(len(cand), np.float32)
        quality = idx.quality[cand]
        w_q = W_QUALITY * (4 if intent.soft_quality else 1)
        open_intent = not intent.concepts
        w_dense = (W_DENSE_OPEN if open_intent else W_DENSE) if self.dense is not None and SEARCH_MODE != 'lexical' else 0.0
        w_lex = 0.0 if SEARCH_MODE == 'dense' and self.dense is not None else (
            W_LEX_OPEN if open_intent and w_dense else W_LEX)
        if SEARCH_MODE == 'lexical':
            parts['dense'] = parts['lexical']
        score = (w_dense * parts['dense'] + w_lex * parts['lexical'] + W_CONCEPT * parts['concept']
                 + W_PATTERN * parts['pattern'] + W_OCCASION * parts['occasion'] + w_q * quality)
        if intent.who is None:
            score -= CHILD_PRIOR * idx.who['kids'][cand]
        if intent.colors:
            score += W_EXACT_COLOR * idx.color_mask(intent.colors[:1])[cand]
        if intent.concepts:
            primary_types = {t for c in intent.concepts for t in CONCEPTS[c][0]}
            if primary_types:
                score += W_PRIMARY_TYPE * np.isin(idx.codes['product_type'][cand],
                                                  [idx.vocab['product_type'][t.casefold()] for t in primary_types
                                                   if t.casefold() in idx.vocab['product_type']])
        for phrase in intent.demote_phrases:
            score -= DEMOTE_PENALTY * idx.phrase_mask(phrase)[cand]
        if intent.soft_cheap or intent.soft_premium:
            price = idx.price[cand]
            rel = (price - price.min()) / max(price.max() - price.min(), 1)
            score += 0.12 * ((1 - rel) if intent.soft_cheap else rel)
        return cand, score.astype(np.float64), parts

    def _rank_intent(self, intent, filters, base, trace, exact_name=()):
        mask, concept_strength, notes = self._intent_mask(intent, filters, base)
        if len(exact_name):
            # named products join the candidates; explicit filters still apply to them
            mask |= np.isin(np.arange(self.index.n), exact_name) & base
        cand, score, parts = self._score(intent, mask, concept_strength)
        if len(exact_name) and len(cand):
            score = score + np.isin(cand, exact_name)   # the named product(s) lead the list
        if not len(cand):
            if trace is not None:
                trace.setdefault('intents', []).append({'text': intent.text, 'candidates': 0, 'notes': notes})
            return {'ranked': cand, 'score': score, 'intent': intent}
        # Deterministic: score desc, then ID asc (positions are in ID order).
        order = np.lexsort((cand, -np.round(score, 9)))
        ranked, ranked_score = cand[order], score[order]
        ranked, ranked_score = self._diversify(ranked, ranked_score)
        if self.reranker is not None and RERANK_WEIGHT > 0:
            ranked, ranked_score = self._rerank(intent, ranked, ranked_score)
        if not intent.concepts and self.dense is not None and 'dense_raw' in parts:
            lookup = np.zeros(self.index.n, dtype=np.float32)
            lookup[cand] = parts['dense_raw']
            ranked, ranked_score = self._semantic_cutoff(ranked, ranked_score, lookup[ranked])
        if trace is not None:
            top = ranked[:10]
            pos_in = {p: k for k, p in enumerate(cand)}
            trace.setdefault('intents', []).append({
                'text': intent.text, 'candidates': int(len(cand)), 'notes': notes,
                'top': [{'id': self.index.ids[p], 'score': round(float(ranked_score[k]), 4),
                         **{name: round(float(vals[pos_in[p]]), 3) for name, vals in parts.items()
                            if name != 'dense_raw'}} for k, p in enumerate(top)]})
        return {'ranked': ranked[:MAX_CANDIDATES], 'score': ranked_score[:MAX_CANDIDATES], 'intent': intent}

    def _rerank(self, intent, ranked, score):
        """Reorder only the head of an already-filtered list; membership and the tail are unchanged."""
        from .dense import document_text
        k = min(RERANK_TOP, len(ranked))
        if k < 2:
            return ranked, score
        head = ranked[:k]
        with self._lock:
            ce = np.array(list(self.reranker.rerank(intent.semantic_text,
                                                    [document_text(self.products[i]) for i in head])), dtype=np.float64)
        norm = lambda x: (x - x.min()) / max(x.max() - x.min(), 1e-9)
        mixed = (1 - RERANK_WEIGHT) * norm(score[:k]) + RERANK_WEIGHT * norm(ce)
        order = np.lexsort((head, -np.round(mixed, 9)))
        # keep the head's scores in descending order so later thresholds still see a sorted list
        return np.concatenate([head[order], ranked[k:]]), np.concatenate([np.sort(score[:k])[::-1], score[k:]])

    def _diversify(self, ranked, score, window=60):
        """Soften runs of the same product name (colour variants) inside the visible head of the list."""
        head = min(window, len(ranked))
        if head < 3:
            return ranked, score
        seen = {}
        adj = score[:head].copy()
        for k in range(head):
            name = self.products[ranked[k]]['name'].casefold()
            adj[k] -= NAME_REPEAT_PENALTY * seen.get(name, 0)
            seen[name] = seen.get(name, 0) + 1
        order = np.lexsort((ranked[:head], -np.round(adj, 9)))
        return np.concatenate([ranked[:head][order], ranked[head:]]), np.concatenate([adj[order], score[head:]])

    def _semantic_cutoff(self, ranked, score, dense_raw):
        """Without a garment anchor, keep only the semantically credible head of the list."""
        best = float(dense_raw.max())
        keep = dense_raw >= best - 0.12
        keep[:10] = True
        return ranked[keep], score[keep]

    def _merge(self, intents_ranked, sort_by, sort_order):
        lists = [r['ranked'] for r in intents_ranked if len(r['ranked'])]
        if not lists:
            return np.zeros(0, dtype=np.int64)
        if sort_by == 'relevance':
            merged = _interleave(lists)
            return merged[::-1] if sort_order == 'asc' else merged
        # Field sort: keep the relevant head of each choice, then sort exactly by the field.
        keep = []
        for r in intents_ranked:
            ranked, score = r['ranked'], r['score']
            if not len(ranked):
                continue
            if r['intent'].concepts:
                floor = score[0] - (1 - FIELD_SORT_KEEP) * max(score[0], 1e-9)
                head = ranked[score >= floor][:FIELD_SORT_CAP * 5]
            else:
                head = ranked[:max(30, int(np.sum(score >= score[0] * FIELD_SORT_KEEP)))][:FIELD_SORT_CAP]
            keep.append(head)
        union = np.unique(np.concatenate(keep))
        return self.index.sort_subset(union, sort_by, sort_order)

    # ------------------------------------------------------------------ recommendations
    def _recommend(self, page, plan, intents_ranked, filters, k=5):
        idx = self.index
        page_set = set(int(p) for p in page)
        comp = plan.complement if plan else None
        anchors = [int(page[0])]   # a recommendation only scores if its anchor is itself judged relevant
        if plan and plan.mixed:
            # one anchor per choice so each choice gets complements
            anchors = []
            for r in intents_ranked:
                for p in r['ranked'][:3]:
                    if int(p) in page_set and int(p) not in anchors:
                        anchors.append(int(p))
                        break
            anchors = anchors or [int(page[0])]
        global_who = None
        if plan:
            whos = {i.who for i in plan.intents if i.who}
            global_who = whos.pop() if len(whos) == 1 else None
        if comp is not None and comp.concepts:
            targets = {a: list(comp.concepts) for a in anchors}
        else:
            targets = {a: self._complement_concepts(a, plan) for a in anchors}
        comp_intent = comp or Intent(text='')
        if comp_intent.who is None and global_who:
            comp_intent.who = global_who
        pools = []
        for anchor in anchors:
            for rank, concept in enumerate(targets[anchor]):
                cands = self._complement_candidates(anchor, concept, comp_intent, page_set)
                pools.append((rank, anchor, concept, cands))
        # Round-robin across anchors and complement kinds, best-first inside each pool.
        recs, used_ids, used_names = [], set(), set()
        explicit = comp is not None and comp.concepts
        depth = 0
        while len(recs) < k and any(depth < len(c) for _, _, _, c in pools):
            for rank, anchor, concept, cands in sorted(pools, key=lambda t: (t[0], anchors.index(t[1]))):
                if depth >= len(cands) or len(recs) >= k:
                    continue
                pos, harmony = cands[depth]
                name = self.products[pos]['name'].casefold()
                if pos in used_ids or (name in used_names and not explicit):
                    continue
                used_ids.add(pos)
                used_names.add(name)
                recs.append({'id': idx.ids[pos], 'for_product_id': idx.ids[anchor],
                             'reason': self._reason(anchor, pos, concept, harmony)})
            depth += 1
        return recs

    def _complement_concepts(self, anchor, plan):
        idx = self.index
        concept = None
        if plan:
            for intent in plan.intents:
                if intent.concepts and idx.concept_strength[intent.concepts[0]][anchor] > 0:
                    concept = intent.concepts[0]
        if concept is None:
            strengths = {c: idx.concept_strength[c][anchor] for c in COMPLEMENTS if idx.concept_strength[c][anchor] > 0}
            concept = max(strengths, key=lambda c: (strengths[c], c in COMPLEMENTS)) if strengths else None
        if concept in COMPLEMENTS:
            out = COMPLEMENTS[concept]
        else:
            role = CONCEPTS[concept][2] if concept else None
            out = ROLE_COMPLEMENTS.get(role, ['bag', 'scarf', 'sneakers'])
        # drop complements that are the same kind as the anchor itself
        return [c for c in out if idx.concept_strength[c][anchor] == 0][:4]

    def _complement_candidates(self, anchor, concept, comp, page_set, per_pool=12):
        idx = self.index
        mask = idx.concept_strength[concept] >= 0.85
        anchor_who = {w for w in ('men', 'women', 'kids', 'baby') if idx.who[w][anchor]}
        if comp.who:
            mask &= idx.who[comp.who]
        elif anchor_who:
            mask &= np.any([idx.who[w] for w in anchor_who], axis=0)
        if 'baby' not in anchor_who and comp.who != 'baby':
            mask &= ~idx.who['baby']
        if comp.min_price is not None:
            mask &= idx.price >= comp.min_price
        if comp.max_price is not None:
            mask &= idx.price <= comp.max_price
        if comp.min_rating is not None:
            with np.errstate(invalid='ignore'):
                mask &= idx.rating >= comp.min_rating
        if comp.colors:
            mask &= idx.color_mask(comp.colors)
        if comp.exclude_colors:
            mask &= ~idx.color_mask(comp.exclude_colors)
        cand = np.flatnonzero(mask)
        cand = cand[~np.isin(cand, list(page_set))]
        if not len(cand):
            return []
        anchor_color = self.products[anchor]['color']
        harmony = np.array([color_harmony(anchor_color, self.products[c]['color']) for c in cand])
        if self.dense is not None:
            sim = self.dense.matrix[cand] @ self.dense.matrix[anchor]
            if comp.semantic_text:
                sim = 0.5 * sim + 0.5 * self.dense.scores(comp.semantic_text)[cand]
            sim = (sim - sim.min()) / max(sim.max() - sim.min(), 1e-9)
        else:
            sim = np.zeros(len(cand))
        strength = idx.concept_strength[concept][cand]
        score = 0.45 * sim + 0.35 * harmony + 0.15 * strength + 0.05 * idx.quality[cand]
        order = np.lexsort((cand, -np.round(score, 9)))[:per_pool]
        return [(int(cand[o]), float(harmony[o])) for o in order]

    def _reason(self, anchor, pos, concept, harmony):
        a, c = self.products[anchor], self.products[pos]
        if c['color'] == a['color']:
            tone = f"matches its {a['color'].lower()} colour"
        elif c['color'] in NEUTRALS:
            tone = f"a versatile {c['color'].lower()} that pairs with {a['color'].lower()}"
        elif harmony >= 0.6:
            tone = f"{c['color'].lower()} harmonises with {a['color'].lower()}"
        else:
            tone = f"adds a {c['color'].lower()} accent"
        text = (f"{c['product_type']} ({c['name']}) to complete the look with the {a['product_type'].lower()} "
                f"'{a['name']}'; {tone}.")
        return text[:300]


def _interleave(lists):
    """Fair round-robin across choices so every requested choice appears near the top."""
    out, seen = [], set()
    longest = max(len(l) for l in lists)
    for k in range(longest):
        for l in lists:
            if k < len(l):
                p = int(l[k])
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        if len(out) >= MAX_CANDIDATES:
            break
    return np.array(out, dtype=np.int64)


def _hue(color):
    words = color.lower().replace('greenish khaki', 'khaki').replace('yellowish brown', 'brown').split()
    for w in reversed(words):
        if w in HUE:
            return HUE[w]
    return None


def color_harmony(a, b):
    """Colour-wheel compatibility in [0, 1]: same > neutral > analogous > complementary > clash."""
    if a == b:
        return 1.0
    if a in NEUTRALS or b in NEUTRALS:
        return 0.85
    ha, hb = _hue(a), _hue(b)
    if ha is None or hb is None:
        return 0.5
    d = abs(ha - hb) % 360
    d = min(d, 360 - d)
    if d <= 40:
        return 0.75
    if 150 <= d <= 210:
        return 0.6
    return 0.3
