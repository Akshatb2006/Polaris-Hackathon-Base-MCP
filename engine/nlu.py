"""Query understanding: turn a shopping sentence into a structured, explainable plan.

Pipeline: normalise -> spell-correct against the catalog vocabulary -> split off a
complementary request ("with a matching bag") -> split mixed choices ("jeans and
t-shirts") -> extract per-choice constraints -> scope constraints (per choice vs global)
-> judge whether the request is a shopping request this catalog can serve at all.
"""
import re
from dataclasses import dataclass, field

from .index import analyze, name_key, raw_tokens, text_concepts
from .taxonomy import (BRANDS, CONCEPTS, COLORS, EXPANSIONS, GENERIC_FASHION, MATERIALS, OCCASIONS, OUT_OF_DOMAIN,
                       PATTERNS, STOPWORDS, WHO_WORDS)

WOMEN_ONLY_CONCEPTS = {'dress', 'skirt', 'heels', 'blouse', 'bra', 'bikini', 'tights', 'leggings'}
MEN_ONLY_CONCEPTS = {'tie'}
ACCESSORY_ROLES = {'bag', 'shoes', 'jewellery', 'headwear', 'eyewear', 'neckwear', 'handwear', 'belt', 'hair'}
NUM = r'(\d+(?:\.\d+)?)'
CUR = r'\s*(?:(?:rs\.?|inr|₹|rupees?)\s*)?'
MAX_RE = re.compile(r'(?:under|below|beneath|less than|lesser than|within|upto|up to|max(?:imum)?(?: of)?|'
                    r'not more than|no more than|cheaper than|at most|<=?|budget(?: of| is)?|till|until)'
                    + CUR + NUM)
MIN_RE = re.compile(r'(?:over|above|more than|greater than|at least|min(?:imum)?(?: of)?|starting(?: at| from)?|'
                    r'>=?|costlier than|pricier than)' + CUR + NUM)
RANGE_RE = re.compile(r'(?:between|from)?' + CUR + NUM + CUR + r'\s*(?:-|to|and)\s*' + CUR + NUM
                      + r'(?=\s*(?:rs|inr|₹|rupees|\b(?!pocket|piece|pack|year|month)))')
AROUND_RE = re.compile(r'(?:around|about|approx(?:imately)?|roughly|near|~)' + CUR + NUM)
BARE_PRICE_RE = re.compile(r'(?:rs\.?|inr|₹)\s*' + NUM + r'|' + NUM + r'\s*(?:rs|inr|rupees?)\b')
RATING_RE = re.compile(r'(?:rated|rating|stars?)\s*(?:of\s*)?(?:at least|above|over|>=?)?\s*(\d(?:\.\d)?)\s*(?:\+|stars?)?'
                       r'|(\d(?:\.\d)?)\s*(?:\+\s*)?stars?(?: and above| or more| or above|\+| plus)?')
NEGATION_RE = re.compile(r"\b(?:not|no|without|except|avoid|anything but|non|don't want|dont want)\s+"
                         r"((?:[a-z\-]+)(?:\s+[a-z\-]+)?)")
COMPLEMENT_MARK = re.compile(
    r'\s*(?:,\s*)?\b(?:with|and|plus|\+|along with|together with|paired with)\s+(?:a\s+|an\s+|some\s+|the\s+)?'
    r'(?:matching|coordinating|co-ordinating|complementary|complementing|complimentary|accompanying|'
    r'suitable|appropriate|nice)\b\s*')
COMPLEMENT_GENERIC = re.compile(r'\b(?:complete the look|accessori[sz]e(?: it)?|style it with|'
                                r'and accessories to match|with accessories)\b')
CONTEXT_RE = re.compile(r'\b(?:to go with|to wear with|to match(?: with)?|that goes with|that go with|'
                        r'that match(?:es)?|to pair with|which goes with|for my|to complement|'
                        r'(?:to wear |to go |to put )?(?:over|on top of|under|underneath|beneath) my)\b')
IMPERATIVE_RE = re.compile(r'^\s*(?:calculate|compute|write|code|program|explain|translate|summari[sz]e|define|solve|'
                           r'repair|fix|install|debug|convert|generate|draw|sing|play|cook|bake|book|reserve|'
                           r'schedule|calculate|predict|measure|compare prices of|tell me)\b')
SPLIT_RE = re.compile(r'\s*(?:,|;|\band also\b|\band\b|\bor\b|&|\bplus\b|\bas well as\b|\balong with\b|/)\s*')
SOFT_CHEAP = re.compile(r'\b(?:cheap|cheapest|affordable|inexpensive|budget|economical|low[- ]cost|bargain|value)\b')
SOFT_PREMIUM = re.compile(r'\b(?:premium|luxury|luxurious|expensive|high[- ]end|designer|fancy|upscale)\b')
SOFT_QUALITY = re.compile(r'\b(?:highly rated|top rated|top-rated|best rated|well reviewed|best reviewed|'
                          r'popular|bestseller|best seller|best-selling|most loved|good reviews|reviewed)\b')
EACH_RE = re.compile(r'\b(?:each|both|apiece|per item|every one|all of them|per piece)\b')
INJECTION_RE = re.compile(r'\b(?:ignore (?:all |the |any )?(?:previous|prior|above)|system prompt|you are now|'
                          r'disregard|jailbreak|reveal|api key|password|developer mode|print (?:all|every)|'
                          r'drop table|select \*|union select|insert into|delete from|or 1 ?= ?1|<script|'
                          r'return (?:all|every)(?: the)? (?:ids|products|items))\b|;\s*--')
QUESTION_RE = re.compile(r"^(?:who|whom|whose|when|where|why|which team|how many|how much|how do|how does|how to|"
                         r"what is|what's|what are|what was|is it|are there|can you|could you)\b")


@dataclass
class Intent:
    text: str
    concepts: list = field(default_factory=list)       # hard garment kind (any of)
    colors: list = field(default_factory=list)         # hard catalog colours (any of)
    exclude_colors: list = field(default_factory=list)
    exclude_concepts: list = field(default_factory=list)
    patterns: list = field(default_factory=list)       # soft catalog patterns
    exclude_patterns: list = field(default_factory=list)
    occasion_concepts: list = field(default_factory=list)
    occasion_weights: dict = field(default_factory=dict)
    demote_phrases: list = field(default_factory=list)  # soft penalty, e.g. bridal gowns for a wedding guest
    brand: str = None                                   # hard: product name must carry this brand
    expansion: str = ''
    min_price: float = None
    max_price: float = None
    min_rating: float = None
    who: str = None
    soft_cheap: bool = False
    soft_premium: bool = False
    soft_quality: bool = False
    context: str = ''                                  # "to go with my black dress" -> context only

    @property
    def semantic_text(self):
        return ' '.join(t for t in (self.text, self.expansion) if t).strip()

    def lexical_terms(self):
        # Situation words ("cold", "office") are interpreted via OCCASIONS; matched literally they
        # retrieve "cold shoulder" tops, so only their curated expansion reaches the lexical channel.
        words = [w for w in raw_tokens(self.text)
                 if w not in STOPWORDS and w not in WHO_WORDS and w not in OCCASIONS and not w.isdigit()]
        words += raw_tokens(self.expansion)
        return analyze(' '.join(words))

    def describe(self):
        out = {'text': self.text}
        for key in ('concepts', 'colors', 'exclude_colors', 'exclude_concepts', 'patterns', 'exclude_patterns',
                    'occasion_concepts', 'demote_phrases'):
            if getattr(self, key):
                out[key] = getattr(self, key)
        for key in ('min_price', 'max_price', 'min_rating', 'who', 'expansion', 'context', 'brand'):
            if getattr(self, key) not in (None, ''):
                out[key] = getattr(self, key)
        for key in ('soft_cheap', 'soft_premium', 'soft_quality'):
            if getattr(self, key):
                out[key] = True
        return out


@dataclass
class Plan:
    original: str
    normalized: str
    corrections: dict
    intents: list
    complement: Intent = None
    complement_requested: bool = False
    shoppable: bool = True
    reason: str = ''
    content_terms: list = field(default_factory=list)
    exact_name: list = field(default_factory=list)     # catalog positions whose name equals the query

    @property
    def mixed(self):
        return len(self.intents) > 1

    def describe(self):
        return {'normalized': self.normalized, 'corrections': self.corrections, 'shoppable': self.shoppable,
                'reason': self.reason, 'mixed': self.mixed, 'intents': [i.describe() for i in self.intents],
                'complement_requested': self.complement_requested,
                'complement': self.complement.describe() if self.complement else None}


def normalize(text):
    text = text.casefold().replace('’', "'").replace('‘', "'").replace('₹', ' rs ')
    text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)                       # 2,000 -> 2000
    text = re.sub(r'\b(\d+(?:\.\d+)?)\s*k\b', lambda m: str(int(float(m.group(1)) * 1000)), text)
    text = re.sub(r'\brs\.', 'rs ', text)
    text = re.sub(r'\bt[\s-]?shirts?\b', lambda m: 't-shirts' if m.group(0).endswith('s') else 't-shirt', text)
    text = re.sub(r'\btees?\b', 't-shirt', text)
    text = re.sub(r"\b(wo)?men s\b", lambda m: (m.group(1) or '') + "men's", text)
    return re.sub(r'\s+', ' ', text).strip()


class QueryParser:
    def __init__(self, index):
        self.index = index

    # ------------------------------------------------------------------ public
    def parse(self, query):
        raw = normalize(query)
        exact = self.index.name_index.get(name_key(query))
        # An exact catalog product name is taken literally: no spelling changes, never declined.
        corrected, corrections = (raw, {}) if exact else self._spell(raw)
        plan = Plan(original=query, normalized=corrected, corrections=corrections, intents=[])
        primary_text, comp_text, comp_generic = self._split_complement(corrected)
        plan.complement_requested = comp_text is not None or comp_generic
        segments = self._split_choices(primary_text)
        plan.intents = [self._intent(seg) for seg in segments] or [self._intent(primary_text)]
        if comp_text is not None:
            plan.complement = self._intent(comp_text)
        elif comp_generic:
            plan.complement = Intent(text='')
        self._scope(plan, corrected)
        plan.shoppable, plan.reason, plan.content_terms = self._judge(plan, corrected, raw)
        if exact:
            plan.exact_name = exact
            plan.shoppable, plan.reason = True, 'exact catalog product name'
        return plan

    # ------------------------------------------------------------------ steps
    def _spell(self, text):
        corrections = {}

        def fix(m):
            word = m.group(0)
            fixed = self.index.correct(word)
            if fixed != word:
                corrections[word] = fixed
            return fixed
        return re.sub(r"[a-z]+", fix, text), corrections

    def _split_complement(self, text):
        m = COMPLEMENT_MARK.search(text)
        if m and self._concepts(text[m.end():]):
            return text[:m.start()].strip(), text[m.end():].strip(), False
        g = COMPLEMENT_GENERIC.search(text)
        if g:
            return (text[:g.start()] + ' ' + text[g.end():]).strip(), None, True
        # "a dress with a bag": `with` + accessory concept after a garment concept.
        for m in re.finditer(r'\b(?:with|plus|and also)\s+(?:a |an |some |the )?', text):
            before, after = text[:m.start()], text[m.end():]
            head_after = self._concepts(after.split(' and ')[0])
            head_before = self._concepts(before)
            if head_after and head_before and CONCEPTS[head_after[0]][2] in ACCESSORY_ROLES \
                    and CONCEPTS[head_after[0]][2] != CONCEPTS[head_before[-1]][2] \
                    and not re.match(r'(?:a |an )?(?:tie|belt|hood|bow)\b(?!\s+(?:bag|shoes))', after) :
                return before.strip(), after.strip(), False
        return text, None, False

    def _split_choices(self, text):
        pieces = [p for p in SPLIT_RE.split(text) if p and p.strip()]
        segments, pending = [], []
        for piece in pieces:
            head = CONTEXT_RE.split(piece)[0]
            if self._concepts(head):
                segments.append(' '.join(pending + [piece]).strip())
                pending = []
            elif segments and not self._colors_only(piece):
                segments[-1] = segments[-1] + ' and ' + piece
            else:
                pending.append(piece)
        if pending:
            if segments:
                segments[-1] = segments[-1] + ' ' + ' '.join(pending)
            else:
                segments.append(' '.join(pending))
        # Colour lists like "black and white t-shirts" belong to one choice.
        return segments if len(segments) > 1 else [text]

    def _colors_only(self, piece):
        words = [w for w in raw_tokens(piece) if w not in STOPWORDS]
        return bool(words) and all(any(w in raw_tokens(c) for c in COLORS) for w in words)

    def _concepts(self, text):
        return text_concepts(text)

    def _intent(self, segment):
        intent = Intent(text=segment)
        text = segment
        # Negations first so "not black" never becomes a colour requirement.
        for m in NEGATION_RE.finditer(text):
            phrase = m.group(1)
            for word in (phrase, phrase.split()[0]):
                if word in COLORS:
                    intent.exclude_colors += COLORS[word]
                    break
                if word in PATTERNS:
                    intent.exclude_patterns += PATTERNS[word]
                    break
                cs = self._concepts(word)
                if cs:
                    intent.exclude_concepts += cs
                    break
            if phrase.startswith(('sleeve', 'sleeves')):
                intent.expansion += ' sleeveless'
        positive = NEGATION_RE.sub(' ', text)
        parts = CONTEXT_RE.split(positive, maxsplit=1)
        main, context = (parts[0], parts[1]) if len(parts) > 1 else (positive, '')
        concepts_main = self._concepts(main)
        # "a layer to wear over my shirt": the shirt is context, not what is wanted
        concepts_all = concepts_main or ([] if context.strip() else self._concepts(positive))
        if concepts_all:
            head = concepts_all[-1]
            # Keep sibling concepts named in the same breath ("hoodie or sweatshirt" handled by split).
            intent.concepts = [head]
        if context and concepts_main:
            intent.context = context.strip()
        intent.text = (main if concepts_main else positive).strip()
        colour_zone = main if concepts_main else positive
        intent.colors = self._colors(colour_zone)
        intent.patterns = sorted({v for k, vals in PATTERNS.items() if _has(colour_zone, k) for v in vals})
        intent.expansion = (intent.expansion + ' ' + ' '.join(
            v for k, v in EXPANSIONS.items() if _has(positive, k))).strip()
        occ_concepts, occ_text, occ_hits = [], [], []
        for word, (concepts, expansion) in OCCASIONS.items():
            if _has(positive, word):
                occ_hits.append(concepts)
                occ_concepts += [c for c in concepts if c not in occ_concepts]
                occ_text.append(expansion)
        # concepts shared by several situations ("cold" + "office" -> blazer, coat) weigh most
        intent.occasion_weights = {c: sum(c in hit for hit in occ_hits) / len(occ_hits) for c in occ_concepts}
        if _has(positive, 'gift') or _has(positive, 'present'):
            for c in ('wallet', 'scarf', 'sweater', 'bag', 'socks', 'tie', 'belt', 'nightwear'):
                if c not in occ_concepts:
                    occ_concepts.append(c)
                    intent.occasion_weights[c] = 1.0
        intent.occasion_concepts = occ_concepts
        if occ_text:
            intent.expansion = (intent.expansion + ' ' + ' '.join(occ_text)).strip()
        # A wedding guest is not the bride: bridal gowns stay findable but rank below guest outfits.
        if _has(positive, 'wedding') and not re.search(r'\b(?:bride|bridal|my (?:own )?wedding|getting married|wedding dress)', positive):
            intent.demote_phrases = ['wedding dress', 'bridal', 'bride', 'wedding gown']
        intent.brand = find_brand(positive)
        intent.who = self._who(positive)
        self._prices(positive, intent)
        rating = RATING_RE.search(positive)
        if rating:
            value = float(rating.group(1) or rating.group(2))
            if 1 <= value <= 5:
                intent.min_rating = value
        intent.soft_cheap = bool(SOFT_CHEAP.search(positive))
        intent.soft_premium = bool(SOFT_PREMIUM.search(positive))
        intent.soft_quality = bool(SOFT_QUALITY.search(positive))
        return intent

    def _colors(self, text):
        found, spans = [], []
        for phrase in sorted(COLORS, key=len, reverse=True):
            for m in re.finditer(r'(?<![a-z\-])%s(?![a-z\-])' % re.escape(phrase), text):
                if any(s <= m.start() < e for s, e in spans):
                    continue
                # "rose gold", "wine bottle"... accept; but "dress shirt"/"blue jeans" fine.
                spans.append((m.start(), m.end()))
                found += [c for c in COLORS[phrase] if c not in found]
        return found

    def _who(self, text):
        hits = []
        for tok in raw_tokens(text):
            if tok in WHO_WORDS:
                hits.append(WHO_WORDS[tok])
        if 'maternity' in text or 'pregnan' in text:
            return 'women'
        if not hits:
            return None
        for specific in ('baby', 'boys', 'girls', 'kids'):   # "baby girl" -> baby; "for my son" -> boys
            if specific in hits:
                if specific == 'kids' and 'son' in raw_tokens(text):
                    return 'boys'
                if specific == 'kids' and 'daughter' in raw_tokens(text):
                    return 'girls'
                return specific
        return hits[0]

    def _prices(self, text, intent):
        rng = RANGE_RE.search(text)
        if rng:
            lo, hi = sorted((float(rng.group(1)), float(rng.group(2))))
            if hi >= 100 and (lo >= 100 or re.search(r'rs|inr|rupee|price|between|budget', text)):
                intent.min_price, intent.max_price = lo, hi
                return
        for m in MAX_RE.finditer(text):
            value = float(m.group(1))
            if value >= 50:
                intent.max_price = min(intent.max_price or value, value)
        for m in MIN_RE.finditer(text):
            value = float(m.group(1))
            if value >= 50 and not re.match(r'\s*(?:stars?|rating)', text[m.end():]):
                intent.min_price = max(intent.min_price or value, value)
        if intent.min_price is None and intent.max_price is None:
            m = AROUND_RE.search(text)
            if m and float(m.group(1)) >= 50:
                v = float(m.group(1))
                intent.min_price, intent.max_price = round(v * 0.75), round(v * 1.25)
            else:
                m = BARE_PRICE_RE.search(text)
                if m:
                    intent.max_price = float(m.group(1) or m.group(2))

    def _scope(self, plan, full_text):
        """Decide which constraints are per-choice and which are shared."""
        intents = plan.intents
        all_intents = intents + ([plan.complement] if plan.complement else [])
        each = bool(EACH_RE.search(full_text))
        # Budgets are per item: one stated budget applies to every item, primary or accompanying,
        # unless an item was given its own ("a dress under 2000 with a bag under 1000").
        budgets = {(i.min_price, i.max_price) for i in all_intents if i.min_price is not None or i.max_price is not None}
        if len(budgets) == 1 or (budgets and each):
            lo, hi = sorted(budgets)[-1]
            for i in all_intents:
                if i.min_price is None and i.max_price is None:
                    i.min_price, i.max_price = lo, hi
        brands = [i.brand for i in intents if i.brand]
        if len(set(brands)) == 1:   # "nike shorts and t-shirts": the brand applies to every choice
            for i in intents:
                i.brand = i.brand or brands[0]
        whos = {i.who for i in all_intents if i.who}
        if len(whos) == 1:
            shared = whos.pop()
            for i in all_intents:
                i.who = i.who or shared
        ratings = [i.min_rating for i in intents if i.min_rating]
        if len(ratings) == 1:
            for i in intents:
                i.min_rating = i.min_rating or ratings[0]
        for flag in ('soft_cheap', 'soft_premium', 'soft_quality'):
            if any(getattr(i, flag) for i in all_intents):
                for i in all_intents:
                    setattr(i, flag, True)
        for i in all_intents:
            self._fit_occasion_to_audience(i)
        occ = [c for i in intents for c in i.occasion_concepts]
        if occ:
            for i in all_intents:
                if not i.occasion_concepts:
                    i.occasion_concepts = list(dict.fromkeys(occ))
                    i.occasion_weights = {c: 1.0 for c in i.occasion_concepts}

    @staticmethod
    def _head_noun(text):
        """Last content word before the first preposition: "office chair for work" -> "chair"."""
        head = HEAD_SPLIT_RE.split(text, maxsplit=1)[0]
        words = [w for w in raw_tokens(head) if w not in STOPWORDS and not w.isdigit() and w.isalpha()]
        return words[-1] if words else None

    @staticmethod
    def _fit_occasion_to_audience(intent):
        """An occasion suggests different garments for different shoppers ("first date ... for him")."""
        if intent.who == 'men':
            drop, words = WOMEN_ONLY_CONCEPTS, ('feminine',)
        elif intent.who == 'women':
            drop, words = MEN_ONLY_CONCEPTS, ('masculine',)
        else:
            return
        intent.occasion_concepts = [c for c in intent.occasion_concepts if c not in drop]
        intent.occasion_weights = {c: w for c, w in intent.occasion_weights.items() if c not in drop}
        intent.expansion = ' '.join(w for w in intent.expansion.split() if w not in words)

    def _judge(self, plan, text, raw):
        """Is this a request the catalog can answer? Returns (shoppable, reason, content_terms)."""
        tokens = raw_tokens(text)
        content = [t for t in tokens if t not in STOPWORDS and not t.isdigit()]
        has_concept = any(i.concepts for i in plan.intents) or (plan.complement and plan.complement.concepts)
        ood = [w for w in OUT_OF_DOMAIN if _has(text, w)]
        if INJECTION_RE.search(text) and not has_concept:
            return False, 'instruction-like text without a shopping request', content
        if has_concept and ood and not self._concepts(raw):
            # the garment only appeared after spelling correction ("car brakes" -> "braces")
            return False, f'asks for items outside a fashion catalog ({", ".join(sorted(ood)[:3])})', content
        if has_concept:
            return True, 'names a catalog garment or accessory', content
        head = self._head_noun(text)
        if head and head not in self.index.lexicon and head not in MATERIALS and head not in GENERIC_FASHION \
                and (self.index.catalog_freq.get(head, 0) == 0 or any(_has(head, w) for w in OUT_OF_DOMAIN)):
            return False, f'the requested thing ("{head}") is not something this catalog sells', content
        if ood and not any(t in GENERIC_FASHION for t in tokens):
            return False, f'asks for items outside a fashion catalog ({", ".join(sorted(ood)[:3])})', content
        signals = [t for t in tokens if t in GENERIC_FASHION or t in WHO_WORDS or t in MATERIALS]
        attrs = any(i.colors or i.patterns or i.occasion_concepts or i.expansion for i in plan.intents)
        if signals or attrs:
            return True, 'fashion attributes, occasion or audience recognised', content
        if not content:
            return False, 'no shopping content', content
        if QUESTION_RE.search(text) or IMPERATIVE_RE.search(text):
            return False, 'an instruction or factual question rather than a shopping request', content
        known = [t for t in content if self.index.df.get(analyze(t)[0] if analyze(t) else t, 0) >= 2]
        if len(known) / len(content) >= 0.6:
            return True, 'query words are well attested in the catalog', content
        return False, 'too little catalog evidence for this request', content


BRAND_RES = {brand: re.compile(r"(?<![a-z0-9])(?:%s)(?![a-z0-9])" % '|'.join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)))
             for brand, aliases in BRANDS.items()}


STYLE_OF_RE = re.compile(r'^\s*(?:style|inspired|type|look|like|dupe|alternative)\b')


def find_brand(text):
    """A brand the shopper requires; "converse style sneakers" or "like nike" describe a look instead."""
    for brand, rx in BRAND_RES.items():
        for m in rx.finditer(text):
            before, after = text[:m.start()], text[m.end():]
            if STYLE_OF_RE.match(after) or re.search(r'\b(?:like|similar to|than|not|no|without|non)\s*$', before):
                continue
            return brand
    return None


HEAD_SPLIT_RE = re.compile(r'\b(?:for|to|on|at|in|with|near|from|during|that|which|so|because|while|like)\b')


def _has(text, phrase):
    return re.search(r'(?<![a-z0-9])%s(?![a-z0-9])' % re.escape(phrase), text) is not None
