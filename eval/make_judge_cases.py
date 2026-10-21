"""Regenerate eval/judge_cases.json deterministically: python eval/make_judge_cases.py

Labels describe what a shopper would accept, written from catalog vocabulary (product types,
colour values, description words) and never from engine output. Cases are grouped by topic and
whole groups go to one split, so near-duplicate intents ("wedding guest", "wedding in December")
never straddle dev and held-out. The held-out list is frozen: do not edit it while tuning.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W = {'audiences': ['Ladieswear', 'Ladies Accessories', 'Lingeries/Tights', 'Divided'], 'sections': ['Ladies H&M Sport']}
M = {'audiences': ['Menswear'], 'sections': ['Men H&M Sport']}
K = {'audiences': ['Children Sizes 92-140', 'Children Sizes 134-170', 'Children Accessories, Swimwear', 'Baby Sizes 50-98'],
     'sections': ['Kids Sports']}
BAG = ['Bag', 'Weekend/Gym bag', 'Backpack', 'Cross-body bag', 'Tote bag', 'Shoulder bag', 'Bumbag']
SHOES = ['Sneakers', 'Boots', 'Bootie', 'Sandals', 'Heeled sandals', 'Flip flop', 'Wedge', 'Pumps', 'Heels',
         'Ballerinas', 'Flat shoe', 'Flat shoes', 'Moccasins', 'Other shoe']
SWIM = ['Swimsuit', 'Bikini top', 'Swimwear bottom', 'Swimwear set', 'Swimwear top']
BLUE = ['Blue', 'Dark Blue', 'Light Blue', 'Other Blue']
RED = ['Red', 'Dark Red', 'Light Red', 'Other Red']
GREEN = ['Green', 'Dark Green', 'Light Green', 'Other Green', 'Greenish Khaki']
WHITE = ['White', 'Off White']
GREY = ['Grey', 'Dark Grey', 'Light Grey']
BEIGE = ['Beige', 'Light Beige', 'Dark Beige', 'Greyish Beige']
PRINT = ['floral', 'flower', 'patterned', 'print']
FESTIVE = ['satin', 'chiffon', 'lace', 'sequin', 'elegant', 'draping', 'pleat', 'velvet', 'velour']
BRIDAL = ['wedding dress', 'bridal', 'bride']


def e(**kw):
    return kw


def c(kind, query, **kw):
    return {'kind': kind, 'query': query, **kw}


# group -> (split, cases). Assign whole topics so wording variants stay on one side.
GROUPS = {
    'tops': ('dev', [
        c('single', 'linen shirt for men', expect=e(types=['Shirt'], prefer=['linen'], **M)),
        c('single', 'striped long sleeve t-shirt', expect=e(types=['T-shirt', 'Top'], prefer=[['stripe'], 'long-sleeved'])),
        c('single', 'turtleneck sweater', expect=e(types=['Sweater'], prefer=[['polo-neck', 'turtleneck', 'polo neck']])),
        c('single', 'oversized grey hoodie', expect=e(types=['Hoodie'], also=['Sweater'], colors=GREY, prefer=[['oversized', 'loose']])),
    ]),
    'bottoms': ('heldout', [
        c('single', 'high waisted wide leg jeans', expect=e(words=['jeans'], also=['Trousers'], prefer=[['high waist', 'high-waisted'], 'wide'])),
        c('single', 'mens slim fit chinos', expect=e(words=['chinos'], also=['Trousers'], prefer=['slim'], **M)),
        c('single', 'pleated mini skirt', expect=e(types=['Skirt'], prefer=['pleat', ['short', 'mini']])),
        c('single', 'running shorts', expect=e(types=['Shorts'], prefer=[['sports', 'running', 'fast-drying', 'functional']])),
    ]),
    'dresses': ('dev', [
        c('single', 'floral midi dress', expect=e(types=['Dress'], words=['dress'], prefer=[PRINT, ['calf-length', 'midi']])),
        c('single', 'satin slip dress', expect=e(types=['Dress'], words=['dress'], prefer=['satin', ['strap', 'slip']])),
        c('single', 'green summer dress under 2000', expect=e(types=['Dress'], words=['dress'], colors=GREEN, max_price=2000,
                                                               prefer=[['airy', 'sleeveless', 'short', 'linen', 'cotton', 'strap']])),
        c('single', 'baby girl dress', expect=e(types=['Dress'], words=['dress'], audiences=['Baby Sizes 50-98'], sections=['Baby Girl'])),
    ]),
    'outerwear': ('heldout', [
        c('single', 'padded winter jacket for women', expect=e(types=['Jacket', 'Coat'], prefer=[['padded', 'puffer', 'quilted'], ['warm', 'lined']], **W)),
        c('single', 'beige trench coat', expect=e(types=['Coat'], also=['Jacket'], colors=BEIGE, prefer=[['trench', 'double-breasted', 'belt']])),
        c('single', 'denim jacket', expect=e(types=['Jacket'], prefer=['denim'])),
        c('single', 'kids rain jacket', expect=e(types=['Jacket', 'Coat', 'Outdoor overall'], prefer=[['water-repellent', 'waterproof', 'rain']], **K)),
    ]),
    'knit-and-tailoring': ('dev', [
        c('single', 'chunky knit cardigan', expect=e(types=['Cardigan'], prefer=[['chunky', 'knit']])),
        c('single', 'black blazer for women', expect=e(types=['Blazer'], colors=['Black'], **W)),
    ]),
    'shoes': ('heldout', [
        c('single', 'black leather ankle boots', expect=e(types=['Boots', 'Bootie'], colors=['Black'], prefer=['leather', ['ankle', 'short']])),
        c('single', 'white sneakers for men under 3000', expect=e(types=['Sneakers'], colors=WHITE, max_price=3000, **M)),
    ]),
    'accessories': ('dev', [
        c('single', 'gold hoop earrings', expect=e(types=['Earring', 'Earrings'], colors=['Gold'], prefer=['hoop'])),
        c('single', 'wool scarf', expect=e(types=['Scarf'], prefer=['wool'])),
        c('single', 'leather belt for men', expect=e(types=['Belt'], prefer=['leather'], **M)),
        c('single', 'crossbody bag black', expect=e(types=BAG, colors=['Black'], prefer=[['shoulder strap', 'crossbody', 'cross-body', 'narrow strap']])),
    ]),
    'accessories-2': ('heldout', [
        c('single', 'bucket hat', expect=e(types=['Hat/brim', 'Bucket hat', 'Hat/beanie'], prefer=['bucket'])),
        c('single', 'silver necklace', expect=e(types=['Necklace'], colors=['Silver'])),
    ]),
    'intimates-and-sleep': ('dev', [
        c('single', 'cotton pyjama set', expect=e(types=['Pyjama set'], words=['pyjama'], also=['Pyjama bottom', 'Pyjama jumpsuit/playsuit'], prefer=['cotton'])),
        c('single', 'red lace bra', expect=e(types=['Bra'], colors=RED, prefer=['lace'])),
    ]),
    'sport-and-swim': ('heldout', [
        c('single', 'sports bra high support', expect=e(types=['Bra'], prefer=[['sports', 'sport'], ['high support', 'firm support', 'maximum support']])),
        c('single', 'swim shorts for men', expect=e(types=['Swimwear bottom', 'Shorts'], prefer=['swim'], **M)),
    ]),
    'typos': ('dev', [
        c('single', 'sneekers for women', expect=e(types=['Sneakers'], **W)),
        c('single', 'blak hodie', expect=e(types=['Hoodie'], also=['Sweater'], colors=['Black'])),
        c('single', 'leopard prnt skirt', expect=e(types=['Skirt'], prefer=[['leopard', 'animal', 'print']])),
    ]),
    'typos-2': ('heldout', [
        c('single', 'trosers for office', expect=e(types=['Trousers'], prefer=[['tailored', 'woven', 'suit', 'crease']])),
        c('single', 'swetar for kids', expect=e(types=['Sweater', 'Cardigan'], **K)),
    ]),
    'budgets': ('dev', [
        c('single', 'jeans between 1000 and 2000', expect=e(words=['jeans'], also=['Trousers'], min_price=1000, max_price=2000)),
        c('single', 'cheap t-shirts under rs 500', expect=e(types=['T-shirt'], max_price=500)),
    ]),
    'budgets-2': ('heldout', [
        c('single', 'blue shirt below 1500', expect=e(types=['Shirt'], colors=BLUE, max_price=1500)),
        c('single', 'dress around 2000', expect=e(types=['Dress'], words=['dress'], min_price=1500, max_price=2500)),
    ]),
    'weddings': ('dev', [
        c('unusual', 'wedding guest outfit', expect=e(types=['Dress', 'Jumpsuit/Playsuit', 'Blazer', 'Blouse', 'Skirt'], prefer=[FESTIVE], avoid=BRIDAL)),
        c('unusual', 'what to wear to a wedding in december', expect=e(types=['Dress', 'Jumpsuit/Playsuit', 'Blazer', 'Coat', 'Jacket'],
                                                                     prefer=[['long-sleeved', 'velvet', 'velour', 'wool', 'lined', 'satin']], avoid=BRIDAL)),
        c('unusual', 'summer wedding in goa', expect=e(types=['Dress', 'Jumpsuit/Playsuit', 'Shirt', 'Skirt', 'Blouse'],
                                                       prefer=[['airy', 'linen', 'chiffon', 'light', 'cotton']], avoid=BRIDAL)),
    ]),
    'comfort-and-weather': ('heldout', [
        c('unusual', 'something comfortable for a lazy sunday', expect=e(types=['Trousers', 'Hoodie', 'Sweater', 'Pyjama set', 'Pyjama bottom', 'Shorts', 'T-shirt'],
                                                                         prefer=[['soft', 'relaxed', 'sweatshirt', 'jersey', 'loose']])),
        c('unusual', 'i am always cold at the office', expect=e(types=['Sweater', 'Cardigan', 'Blazer', 'Jacket', 'Coat', 'Scarf'], prefer=[['wool', 'knit', 'warm', 'fleece', 'lined']])),
        c('unusual', 'something to keep my ears warm', expect=e(types=['Hat/beanie', 'Beanie', 'Hat/brim'], prefer=[['knit', 'warm', 'fleece', 'wool', 'lined']])),
        c('unusual', 'what can my toddler wear in the snow', expect=e(types=['Jacket', 'Outdoor overall', 'Boots', 'Gloves', 'Hat/beanie', 'Coat', 'Outdoor trousers'],
                                                                      prefer=[['padded', 'warm', 'lined', 'water-repellent']], **K)),
        c('unusual', 'clothes for hiking in the rain', expect=e(types=['Jacket', 'Outdoor trousers', 'Boots', 'Coat', 'Outdoor Waistcoat'],
                                                                prefer=[['water-repellent', 'waterproof', 'windproof', 'functional']])),
    ]),
    'work-and-dates': ('dev', [
        c('unusual', 'outfit for a job interview', expect=e(types=['Blazer', 'Shirt', 'Trousers', 'Blouse', 'Skirt', 'Dress'], prefer=[['tailored', 'woven', 'smart', 'suit']])),
        c('unusual', 'first date outfit for him', expect=e(types=['Shirt', 'Blazer', 'Trousers', 'Sweater', 'Polo shirt'], **M)),
        c('unusual', 'work from home but still look presentable', expect=e(types=['Shirt', 'Blouse', 'Sweater', 'Cardigan', 'Top', 'Trousers'], prefer=[['soft', 'relaxed', 'knit', 'jersey']])),
    ]),
    'holidays-and-parties': ('heldout', [
        c('unusual', 'what should i pack for a beach holiday', expect=e(types=SWIM + ['Sarong', 'Shorts', 'Sandals', 'Flip flop', 'Dress', 'Sunglasses', 'Hat/brim'],
                                                                        prefer=[['beach', 'swim', 'airy', 'linen', 'summer']])),
        c('unusual', 'something sparkly for a new years eve party', expect=e(types=['Dress', 'Top', 'Skirt', 'Blouse', 'Jumpsuit/Playsuit'],
                                                                             prefer=[['sequin', 'glitter', 'metallic', 'shimmer', 'lurex']])),
        c('unusual', '90s grunge look', expect=e(types=['Shirt', 'T-shirt', 'Trousers', 'Jacket', 'Dress', 'Boots', 'Top', 'Sweater'],
                                                 prefer=[['check', 'flannel', 'oversized', 'washed', '90s', 'grunge', 'distressed']])),
    ]),
    'gym': ('dev', [
        c('unusual', 'gym clothes for women', expect=e(types=['Leggings/Tights', 'Bra', 'Top', 'Shorts', 'T-shirt', 'Vest top'],
                                                       prefer=[['sports', 'functional', 'fast-drying', 'training']], **W)),
    ]),
    'brands-and-attributes': ('dev', [
        # "size 42 ..." is deliberately absent: the catalog has no sizes, and whether a size request is
        # impossible or should be ignored is ambiguous, so it is not labelled either way.
        c('empty', 'nike running shoes'), c('empty', 'gucci handbag'),
        c('empty', 'waterproof smartwatch'), c('empty', 'jeans under 500 and over 3000'),
    ]),
    'brands-and-attributes-2': ('heldout', [
        c('empty', 'levis 501 jeans'), c('empty', 'zara dress'), c('empty', 'adidas tracksuit'),
        c('empty', 'dress under 50 rupees'), c('empty', 'silk saree for a wedding'),
    ]),
    'off-topic': ('dev', [c('empty', 'best laptop for coding'), c('empty', 'how to bake a cake')]),
    'off-topic-2': ('heldout', [c('empty', 'book me a flight to dubai'), c('empty', 'what is the capital of france')]),
    'mixed': ('dev', [
        c('mixed', 'black jeans and a white shirt', choices=[e(words=['jeans'], also=['Trousers'], colors=['Black']), e(types=['Shirt', 'Blouse'], colors=WHITE)]),
        c('mixed', 'a red dress and gold earrings', choices=[e(types=['Dress'], words=['dress'], colors=RED), e(types=['Earring', 'Earrings'], colors=['Gold'])]),
        c('mixed', 'blazer and trousers for the office under 3000 each', choices=[e(types=['Blazer'], max_price=3000), e(types=['Trousers'], max_price=3000)]),
        c('mixed', 'winter coat and boots for women', choices=[e(types=['Coat', 'Jacket'], **W), e(types=['Boots', 'Bootie'], **W)]),
    ]),
    'mixed-2': ('heldout', [
        c('mixed', 'sneakers or sandals for summer', choices=[e(types=['Sneakers']), e(types=['Sandals', 'Heeled sandals', 'Flip flop'])]),
        c('mixed', 'hoodie, joggers and a cap for my son', choices=[e(types=['Hoodie', 'Sweater'], **K), e(types=['Trousers'], words=['joggers'], **K),
                                                                   e(types=['Cap/peaked', 'Cap', 'Hat/beanie', 'Hat/brim'], **K)]),
        c('mixed', 'bikini and a beach cover up', choices=[e(types=['Bikini top', 'Swimwear bottom', 'Swimwear set']), e(types=['Sarong'], words=['kaftan', 'cover-up', 'beach'])]),
        c('mixed', 'navy polo shirt or a white t-shirt for men', choices=[e(types=['Polo shirt'], colors=['Dark Blue'], **M), e(types=['T-shirt'], colors=WHITE, **M)]),
    ]),
    'recs': ('dev', [
        c('recs', 'a dress for dinner under 2000 with a matching bag under 1000', args={'include_recommendations': True},
          expect=e(types=['Dress'], words=['dress'], max_price=2000), rec_expect=e(types=BAG, max_price=1000)),
        c('recs', 'mens jeans', args={'include_recommendations': True}, expect=e(words=['jeans'], also=['Trousers'], **M),
          rec_expect=e(types=['T-shirt', 'Shirt', 'Sneakers', 'Belt', 'Top', 'Polo shirt', 'Sweater', 'Hoodie'], **M)),
    ]),
    'recs-2': ('heldout', [
        c('recs', 'winter coat for women', args={'include_recommendations': True}, expect=e(types=['Coat', 'Jacket'], **W),
          rec_expect=e(types=['Scarf', 'Gloves', 'Hat/beanie', 'Beanie', 'Boots', 'Bootie'], **W)),
        c('recs', 'black suit with matching shoes', args={'include_recommendations': True}, expect=e(types=['Blazer', 'Trousers'], words=['suit']),
          rec_expect=e(types=SHOES)),
    ]),
}


def exact_name_cases():
    products = sorted(json.loads((ROOT / 'data/products.json').read_text()), key=lambda p: p['id'])
    rng = random.Random(20261021)
    picks, seen_types = [], set()
    for p in rng.sample(products, 400):
        if p['product_type'] not in seen_types and len(p['name']) > 6 and p['category'] != 'Unknown':
            seen_types.add(p['product_type'])
            picks.append(c('single', p['name'], expect=e(name=p['name'], type=p['product_type'])))
        if len(picks) == 6:
            break
    return {'names': ('dev', picks[:3]), 'names-2': ('heldout', picks[3:])}


if __name__ == '__main__':
    groups = GROUPS | exact_name_cases()
    cases = []
    for group, (split, items) in groups.items():
        for case in items:
            cases.append({'group': group, 'split': split, **case})
    queries = [x['query'].casefold() for x in cases]
    assert len(queries) == len(set(queries)), 'duplicate query'
    (Path(__file__).parent / 'judge_cases.json').write_text(json.dumps(cases, indent=1))
    from collections import Counter
    print(len(cases), dict(Counter((x['split'], x['kind']) for x in cases)))
