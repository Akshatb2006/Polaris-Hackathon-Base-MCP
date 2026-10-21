"""Fixture tests for query understanding and ranking rules (lexical mode, no model needed)."""
import unittest

from search import Catalog


def product(id, name, ptype, color='Black', price=999, description=None, audience='Ladieswear',
            section='Womens Everyday Collection', category='Garment Upper body', pattern='Solid', rating=4.0, count=2):
    return dict(id=id, name=name, description=description or f'{ptype} in soft jersey.', product_type=ptype,
                category=category, audience=audience, section=section, color=color, pattern=pattern,
                price={'amount': price, 'currency': 'INR'}, rating={'average': rating, 'count': count})


CATALOG = [
    product('001', 'Ava dress', 'Dress', 'Red', 1299, 'Calf-length dress in a woven fabric.', category='Garment Full body'),
    product('002', 'Bea dress', 'Dress', 'Black', 2499, 'Short dress in jersey.', category='Garment Full body'),
    product('003', 'Grace dress', 'Sweater', 'Light Pink', 1499, 'Dress in soft sweatshirt fabric.'),
    product('004', 'Cleo jeans', 'Trousers', 'Black', 1799, '5-pocket jeans in washed denim.', pattern='Denim',
            category='Garment Lower body'),
    product('005', 'Dina tee', 'T-shirt', 'White', 499, 'T-shirt in cotton jersey.'),
    product('006', 'Eve bag', 'Bag', 'Black', 899, 'Small shoulder bag in imitation leather.', category='Accessories',
            audience='Ladies Accessories', section='Womens Big accessories'),
    product('007', 'Fay bag', 'Bag', 'Gold', 1999, 'Clutch bag in metallic fabric.', category='Accessories',
            audience='Ladies Accessories', section='Womens Big accessories'),
    product('008', 'Gus sneaker', 'Sneakers', 'White', 2999, 'Trainers in imitation leather.', category='Shoes',
            audience='Menswear', section='Men Shoes'),
    product('009', 'Tea sock', 'Socks', 'Black', 199, 'Trainer socks in a cotton blend.', category='Socks & Tights'),
    product('010', 'Mike tee', 'T-shirt', 'Blue', 599, 'T-shirt in cotton jersey.', audience='Menswear',
            section='Contemporary Casual'),
]


class EngineRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog(CATALOG, dense=None)

    def ids(self, **args):
        return [r['id'] for r in self.catalog.search(args)['results']]

    def test_budget_without_currency_word(self):
        self.assertEqual(self.ids(query='dress under 1500'), ['001', '003'])

    def test_garment_recovered_from_description_head(self):
        self.assertIn('003', self.ids(query='dress'))          # typed Sweater, described as a dress

    def test_head_final_noun(self):
        self.assertNotIn('009', self.ids(query='sneakers'))    # "trainer socks" are socks

    def test_explicit_filter_beats_query_colour(self):
        self.assertEqual(set(self.ids(query='red dress', filters={'color': 'Black'})), {'002'})

    def test_mixed_choices_cover_both(self):
        got = self.ids(query='black jeans and white t-shirts')
        self.assertEqual(set(got[:2]), {'004', '005'})

    def test_complement_goes_to_recommendations_with_own_budget(self):
        out = self.catalog.search({'query': 'a dress under 2000 with a matching bag under 1000',
                                   'include_recommendations': True})
        self.assertNotIn('006', [r['id'] for r in out['results']])
        self.assertEqual([r['id'] for r in out['recommendations']], ['006'])
        self.assertEqual(self.catalog.search({'query': 'a dress with a matching bag'})['recommendations'], [])

    def test_abstains_on_unrelated_requests(self):
        for q in ('spaceship engine', 'office chair', 'who won the cricket world cup', 'drop table products',
                  'how do I fix my car brakes'):
            with self.subTest(q=q):
                self.assertEqual(self.catalog.search({'query': q}), {'results': [], 'recommendations': []})

    def test_spelling_fixes_typos_but_not_product_names(self):
        self.assertEqual(self.catalog.engine.index.correct('sneekers'), 'sneakers')
        self.assertEqual(self.ids(query='Mike tee')[0], '010')   # exact name, never "bike"

    def test_audience_from_query_is_hard(self):
        self.assertEqual(self.ids(query='t-shirt for men'), ['010'])

    def test_empty_query_sorts_exactly(self):
        self.assertEqual(self.ids(sort_by='price', sort_order='asc', limit=3), ['009', '005', '010'])


if __name__ == '__main__':
    unittest.main()
