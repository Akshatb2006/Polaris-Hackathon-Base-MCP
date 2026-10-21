"""Columnar catalog index: exact filters, presorted field orders, BM25F, concepts, typo vocabulary."""
import math
import re
from collections import Counter, defaultdict

import numpy as np

from .taxonomy import (BRANDS, CONCEPTS, COLORS, EXPANSIONS, GENERIC_FASHION, OCCASIONS, OUT_OF_DOMAIN,
                       PATTERN_CONCEPTS, PATTERNS, REFINES, STOPWORDS, WHO_WORDS)

STRING_FILTERS = ('category', 'product_type', 'color', 'audience', 'section', 'pattern')
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
# BM25F field weights: the garment noun and the name say what a product *is*;
# the long description mostly says what it is made of.
FIELD_WEIGHTS = {'name': 2.5, 'head': 3.0, 'product_type': 2.0, 'description': 1.0, 'color': 1.5,
                 'pattern': 1.2, 'section': 0.6, 'category': 0.5}
K1, B = 1.2, 0.5


def stem(word):
    if len(word) <= 3 or not word.isalpha():
        return word
    for suffix, repl in (('ies', 'y'), ('sses', 'ss'), ('ches', 'ch'), ('shes', 'sh'), ('xes', 'x')):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)] + repl
    if word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
        return word[:-1]
    return word


def raw_tokens(text):
    return TOKEN_RE.findall(text.casefold().replace('’', "'"))


def analyze(text):
    """Stemmed index terms; hyphenated compounds are kept whole and also split."""
    out = []
    for tok in raw_tokens(text):
        out.append(stem(tok))
        if '-' in tok:
            out.extend(stem(part) for part in tok.split('-') if part)
    return out


def name_key(text):
    return ' '.join(raw_tokens(text))


def description_head(description):
    """The garment noun phrase that opens a description, e.g. 'Calf-length dress' or '5-pocket jeans'.

    Adjective lists before the noun are kept ("Long, sleeveless dress") when cutting at the first
    comma would leave no garment noun at all.
    """
    if not description:
        return ''
    clause = re.split(r'\.| in | with | made | from | that | and a | containing ', description, maxsplit=1)[0]
    head = clause.split(',', 1)[0]
    if ',' in clause and not text_concepts(head.casefold()):
        head = clause
    return ' '.join(head.split()[:6])


def _term_pattern(terms):
    alts = sorted({re.escape(t) for t in terms}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9\-])(?:%s)(?:e?s)?(?![a-z0-9])" % '|'.join(alts))


CONCEPT_PATTERNS = {c: _term_pattern(terms) for c, (_, terms, _) in CONCEPTS.items()}
TYPE_TO_CONCEPTS = defaultdict(set)
for _concept, (_types, _, _) in CONCEPTS.items():
    for _t in _types:
        TYPE_TO_CONCEPTS[_t].add(_concept)


def product_who(p):
    aud, sec = p['audience'], p['section']
    who = set()
    if aud == 'Baby Sizes 50-98' or sec.startswith('Baby'):
        who |= {'baby', 'kids'}
    if aud.startswith('Children') or sec.startswith(('Kids', 'Young', 'Girls', 'Boys')) or 'baby' in who:
        who.add('kids')
        if sec in ('Baby Boy', 'Kids Boy', 'Young Boy', 'Boys Underwear & Basics'):
            who.add('boys')
        elif sec in ('Baby Girl', 'Kids Girl', 'Young Girl', 'Girls Underwear & Basics'):
            who.add('girls')
        else:
            who |= {'boys', 'girls'}
    elif aud == 'Menswear' or sec == 'Men H&M Sport':
        who.add('men')
    elif aud in ('Ladieswear', 'Ladies Accessories', 'Lingeries/Tights', 'Divided') or sec == 'Ladies H&M Sport':
        who.add('women')
    return who


def text_concepts(text):
    """Concepts named in a garment phrase, in order; overlapping matches keep the longest term."""
    hits = []
    for concept, pattern in CONCEPT_PATTERNS.items():
        for m in pattern.finditer(text):
            hits.append((m.start(), -(m.end() - m.start()), concept, m.end()))
    hits.sort()
    out, covered_until = [], -1
    for start, _, concept, end in hits:
        if start >= covered_until:
            out.append(concept)
            covered_until = end
    return out


INTIMATE_CATEGORIES = {'Underwear', 'Nightwear', 'Underwear/nightwear', 'Swimwear', 'Socks & Tights'}
INTIMATE_ROLES = {'underwear', 'night', 'swim', 'legwear'}


def product_concepts(p):
    """concept -> evidence strength.

    English garment phrases are head-final ("trainer socks" are socks), so the last
    concept in the description head / name is the product; earlier ones are modifiers.
    """
    found = {}
    head = description_head(p.get('description')).casefold()
    name = p['name'].casefold()
    for concept in TYPE_TO_CONCEPTS.get(p['product_type'], ()):
        found[concept] = 0.9
    extra = PATTERN_CONCEPTS.get(p['pattern'], {}).get(p['product_type'])
    if extra:
        found[extra] = max(found.get(extra, 0), 0.9)
    for text, top in ((head, 1.0), (name, 0.8)):
        named = text_concepts(text)
        for k, concept in enumerate(named):
            strength = top if k == len(named) - 1 else 0.4
            found[concept] = max(found.get(concept, 0), strength)
    if p['category'] in INTIMATE_CATEGORIES:
        # pyjama shorts are not shorts; swim shorts are not shorts for a city outfit
        for concept in list(found):
            if CONCEPTS[concept][2] not in INTIMATE_ROLES and concept not in TYPE_TO_CONCEPTS.get(p['product_type'], ()):
                found[concept] = min(found[concept], 0.5)
    for child, parent in REFINES.items():
        if child in found and parent not in found:
            found[parent] = found[child] * 0.95
    return found


class CatalogIndex:
    def __init__(self, products):
        self.products = sorted(products, key=lambda p: p['id'])
        self.n = n = len(self.products)
        self.ids = [p['id'] for p in self.products]
        self.pos = {pid: i for i, pid in enumerate(self.ids)}
        self.price = np.array([p['price']['amount'] for p in self.products], dtype=np.float64)
        self.rating = np.array([np.nan if p['rating']['average'] is None else p['rating']['average']
                                for p in self.products], dtype=np.float64)
        self.count = np.array([p['rating']['count'] for p in self.products], dtype=np.int64)
        self.codes, self.vocab = {}, {}
        for key in STRING_FILTERS:
            values = [str(p.get(key) or '').strip().casefold() for p in self.products]
            table = {v: i for i, v in enumerate(sorted(set(values)))}
            self.vocab[key] = table
            self.codes[key] = np.array([table[v] for v in values], dtype=np.int32)
        self.display = {key: {str(p.get(key) or '').strip().casefold(): p.get(key) for p in self.products}
                        for key in STRING_FILTERS}
        self.name_index = defaultdict(list)
        for i, p in enumerate(self.products):
            self.name_index[name_key(p['name'])].append(i)
        self._orders = {}
        self._build_semantics()
        self._build_bm25()
        self._build_typo_vocab()

    # ------------------------------------------------------------------ exact path
    def filter_mask(self, filters):
        mask = np.ones(self.n, dtype=bool)
        for key in STRING_FILTERS:
            if key in filters:
                code = self.vocab[key].get(filters[key].strip().casefold())
                if code is None:
                    return np.zeros(self.n, dtype=bool)
                mask &= self.codes[key] == code
        if 'min_price' in filters:
            mask &= self.price >= filters['min_price']
        if 'max_price' in filters:
            mask &= self.price <= filters['max_price']
        if 'min_rating' in filters:
            with np.errstate(invalid='ignore'):
                mask &= ~np.isnan(self.rating) & (self.rating >= filters['min_rating'])
        if 'min_review_count' in filters:
            mask &= self.count >= filters['min_review_count']
        return mask

    def order(self, sort_by, sort_order):
        """Full catalog permutation for a field sort; ties by ID ascending, null ratings last."""
        key = (sort_by, sort_order)
        if key not in self._orders:
            idx = np.arange(self.n)
            sign = -1.0 if sort_order == 'desc' else 1.0
            if sort_by in ('relevance', 'id'):
                order = idx if (sort_by == 'relevance' or sort_order == 'asc') else idx[::-1].copy()
            elif sort_by == 'price':
                order = np.lexsort((idx, sign * self.price))
            elif sort_by == 'review_count':
                order = np.lexsort((idx, sign * self.count))
            else:
                nulls = np.isnan(self.rating)
                order = np.lexsort((idx, sign * np.nan_to_num(self.rating), nulls))
            self._orders[key] = order
        return self._orders[key]

    def sort_subset(self, positions, sort_by, sort_order):
        """Exact field sort of a candidate subset using the global presorted rank."""
        order = self.order(sort_by, sort_order)
        if not hasattr(self, '_rank_cache'):
            self._rank_cache = {}
        rank = self._rank_cache.get((sort_by, sort_order))
        if rank is None:
            rank = np.empty(self.n, dtype=np.int64)
            rank[order] = np.arange(self.n)
            self._rank_cache[(sort_by, sort_order)] = rank
        positions = np.asarray(positions, dtype=np.int64)
        return positions[np.argsort(rank[positions], kind='stable')]

    # ------------------------------------------------------------------ semantics
    def _build_semantics(self):
        n = self.n
        self.concept_strength = {c: np.zeros(n, dtype=np.float32) for c in CONCEPTS}
        self.who = {w: np.zeros(n, dtype=bool) for w in ('men', 'women', 'kids', 'boys', 'girls', 'baby')}
        self.color_values = np.array([p['color'] for p in self.products], dtype=object)
        self.heads = []
        for i, p in enumerate(self.products):
            for concept, strength in product_concepts(p).items():
                self.concept_strength[concept][i] = strength
            for w in product_who(p):
                self.who[w][i] = True
            self.heads.append(description_head(p.get('description')))
        # The garment noun each description opens with ("Top in sweatshirt fabric" is a top even when
        # typed Sweater); '' when the description names no garment.
        self.head_concept = np.array([(text_concepts(h.casefold()) or [''])[-1] for h in self.heads], dtype=object)
        self.who['unknown'] = ~(self.who['men'] | self.who['women'] | self.who['kids'])
        # Uninformative names ("id", "1", "x") give a shopper nothing to go on; rank them last.
        self.weak_name = np.array([len(re.sub(r'[^a-z]', '', p['name'].casefold())) < 3 for p in self.products])
        q = np.where(np.isnan(self.rating), 3.0, self.rating)
        # Bayesian-shrunk rating prior in [-1, 1]; only ever a tie-breaker after relevance.
        self.quality = ((q * self.count + 3.6 * 3) / (self.count + 3) - 3.6) / 1.4

    def color_mask(self, catalog_colors):
        wanted = set(catalog_colors)
        if not hasattr(self, '_color_masks'):
            self._color_masks = {}
        key = frozenset(wanted)
        if key not in self._color_masks:
            self._color_masks[key] = np.array([c in wanted for c in self.color_values], dtype=bool)
        return self._color_masks[key]

    def phrase_mask(self, phrase):
        """Products whose name or description contains a phrase (cached; used for soft penalties)."""
        if not hasattr(self, '_phrase_masks'):
            self._phrase_masks = {}
        if phrase not in self._phrase_masks:
            rx = re.compile(r'(?<![a-z])%s(?![a-z])' % re.escape(phrase))
            self._phrase_masks[phrase] = np.array(
                [bool(rx.search(f"{p['name']} {p.get('description') or ''}".casefold())) for p in self.products])
        return self._phrase_masks[phrase]

    def brand_mask(self, brand):
        """Products whose *name* carries the brand (brand words in descriptions are not evidence)."""
        if not hasattr(self, '_brand_masks'):
            self._brand_masks = {}
        if brand not in self._brand_masks:
            rx = re.compile(r"(?<![a-z0-9])(?:%s)(?![a-z0-9])" % '|'.join(re.escape(a) for a in BRANDS[brand]))
            self._brand_masks[brand] = np.array([bool(rx.search(p['name'].casefold())) for p in self.products])
        return self._brand_masks[brand]

    def pattern_mask(self, catalog_patterns):
        codes = [self.vocab['pattern'][v.casefold()] for v in catalog_patterns if v.casefold() in self.vocab['pattern']]
        return np.isin(self.codes['pattern'], codes)

    # ------------------------------------------------------------------ BM25F
    def _build_bm25(self):
        tf_rows = []
        lengths = np.zeros(self.n, dtype=np.float32)
        df = Counter()
        for i, p in enumerate(self.products):
            fields = {'name': p['name'], 'head': self.heads[i], 'product_type': p['product_type'],
                      'description': p.get('description') or '', 'color': p['color'], 'pattern': p['pattern'],
                      'section': p['section'], 'category': p['category']}
            tf = Counter()
            for field, text in fields.items():
                weight = FIELD_WEIGHTS[field]
                for term in analyze(text):
                    tf[term] += weight
            tf_rows.append(tf)
            lengths[i] = sum(tf.values())
            df.update(tf.keys())
        avg = float(lengths.mean())
        postings = defaultdict(lambda: ([], []))
        for i, tf in enumerate(tf_rows):
            norm = K1 * (1 - B + B * lengths[i] / avg)
            for term, freq in tf.items():
                docs, weights = postings[term]
                docs.append(i)
                weights.append(freq * (K1 + 1) / (freq + norm))
        self.df = df
        self.postings = {}
        for term, (docs, weights) in postings.items():
            idf = math.log(1 + (self.n - df[term] + 0.5) / (df[term] + 0.5))
            self.postings[term] = (np.array(docs, dtype=np.int32), np.array(weights, dtype=np.float32) * idf)

    def bm25(self, terms):
        scores = np.zeros(self.n, dtype=np.float32)
        for term, boost in Counter(terms).items():
            hit = self.postings.get(term)
            if hit is not None:
                scores[hit[0]] += hit[1] * min(boost, 2)
        return scores

    # ------------------------------------------------------------------ typo vocabulary
    def _build_typo_vocab(self):
        """SymSpell-style single-delete dictionary over the catalog's own vocabulary."""
        freq = Counter()
        for p in self.products:
            freq.update(t for t in raw_tokens(f"{p['name']} {p.get('description') or ''} {p['product_type']} "
                                              f"{p['color']} {p['pattern']}") if t.isalpha())
        lexicon = set()
        for table in (COLORS, PATTERNS, EXPANSIONS, OCCASIONS, WHO_WORDS):
            for phrase in table:
                lexicon.update(raw_tokens(phrase))
        for _, terms, _ in CONCEPTS.values():
            for term in terms:
                lexicon.update(raw_tokens(term))
        lexicon |= GENERIC_FASHION
        self.catalog_freq = dict(freq)
        # out-of-domain words and brand names are recognised as written, never "corrected"
        self.ood_words = ood_words = {w for phrase in OUT_OF_DOMAIN for w in raw_tokens(phrase)} | {
            w for aliases in BRANDS.values() for a in aliases for w in raw_tokens(a)
            if len(w) > 2 and w not in lexicon}   # "gap hoodie" protects "gap", never "hoodie"
        self.lexicon = {w for w in lexicon if w.isalpha() and len(w) >= 3}
        for word in lexicon:
            freq[word] += 500   # shopper vocabulary outranks rare catalog tokens
        # Out-of-domain words are recognised as written but never produced by a correction
        # ("mike" must not become "bike", "ford" must not become "food").
        self.word_freq = {w: c for w, c in freq.items()
                          if (w in lexicon or c >= 150) and 3 <= len(w) <= 18 and w.isalpha() and w not in ood_words}
        self.known_words = set(freq) | STOPWORDS | ood_words
        deletes = defaultdict(list)
        for word in self.word_freq:
            variants = {word} | {word[:i] + word[i + 1:] for i in range(len(word))}
            if word in self.lexicon and len(word) >= 5:
                # distance-2 deletes for shopper vocabulary only ("swetar" -> "sweater")
                variants |= {v[:i] + v[i + 1:] for v in list(variants) for i in range(len(v))}
            for v in variants:
                deletes[v].append(word)
        self.deletes = {k: sorted(set(v)) for k, v in deletes.items()}

    def correct(self, word):
        """Return the most plausible catalog word within edit distance 2, or the word itself.

        Words the catalog uses (three or more times) are never changed, so product names such as
        "Mike tee" survive; a catalog hapax one edit from shopper vocabulary is still corrected.
        """
        if len(word) < 4 or not word.isalpha() or word in STOPWORDS or word in self.lexicon:
            return word
        if word in self.ood_words or word in STOPWORDS:
            return word
        if word in self.known_words and self.catalog_freq.get(word, 0) >= 3:
            # Catalog words are kept as written ("Mike tee") unless they are rare and one edit from
            # a shopper word the catalog uses hundreds of times more ("dres" -> "dress").
            freq = self.catalog_freq[word]
            near = [c for c in self._candidates(word) if c in self.lexicon and _damerau(word, c) == 1
                    and c[0] == word[0] and self.word_freq.get(c, 0) >= max(200, 100 * freq)]
            return min(near, key=lambda c: (-self.word_freq.get(c, 0), c)) if near and freq < 50 else word
        if word in self.known_words:
            # a catalog hapax one edit from shopper vocabulary is most likely a typo ("sunglases")
            near = [c for c in self._candidates(word) if c in self.lexicon and _damerau(word, c) == 1
                    and (len(word) >= 6 or c[0] == word[0])]
            return min(near, key=lambda c: (-self.word_freq.get(c, 0), c)) if near else word
        cands = self._candidates(word)
        best, best_key = word, None
        # Two edits only for long words ("system" must not become "sister"); short words keep
        # their first letter ("live" must not become "olive").
        limit = 1 if len(word) < 6 else 2
        for cand in cands:
            dist = _damerau(word, cand)
            if dist > limit or (len(word) < 6 and cand[0] != word[0]):
                continue
            if dist == 2 and (cand[0] != word[0] or cand[-1] != word[-1]):
                continue
            if True:
                key = (dist, cand not in self.lexicon, cand[0] != word[0], -self.word_freq[cand], cand)
                if best_key is None or key < best_key:
                    best, best_key = cand, key
        return best

    def _candidates(self, word):
        probes = {word} | {word[:i] + word[i + 1:] for i in range(len(word))}
        if len(word) >= 6:
            probes |= {p[:i] + p[i + 1:] for p in list(probes) for i in range(len(p))}
        return {cand for probe in probes for cand in self.deletes.get(probe, ())}


def _damerau(a, b):
    if abs(len(a) - len(b)) > 2:
        return 3
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]
