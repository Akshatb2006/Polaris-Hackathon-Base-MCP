"""The evaluator's arithmetic: nDCG@10 against a fixed ideal of ten grade-3 slots."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'eval'))
import judge  # noqa: E402

IDEAL = sum(7 / math.log2(i + 2) for i in range(10))


def product(**kw):
    base = dict(id='1', name='Ava dress', description='Calf-length dress in satin.', product_type='Dress',
                section='Womens', audience='Ladieswear', color='Black', pattern='Solid', price={'amount': 999})
    return base | kw


class JudgeMaths(unittest.TestCase):
    def test_ndcg_bounds(self):
        self.assertAlmostEqual(judge.ndcg([3] * 10), 1.0)
        self.assertEqual(judge.ndcg([]), 0.0)
        self.assertAlmostEqual(judge.ndcg([3]), 7 / IDEAL)
        self.assertAlmostEqual(judge.ndcg([3] * 20), 1.0)          # only the top 10 count
        self.assertLess(judge.ndcg([0, 3]), judge.ndcg([3, 0]))     # order matters

    def test_grades(self):
        spec = {'types': ['Dress'], 'prefer': ['satin', 'calf-length']}
        self.assertEqual(judge.grade(product(), spec), 3)
        self.assertEqual(judge.grade(product(description='Short dress in jersey.'), spec), 1)
        self.assertEqual(judge.grade(product(product_type='Skirt', description='Skirt.'), spec), 0)

    def test_hard_constraints_force_zero(self):
        spec = {'types': ['Dress'], 'colors': ['Red'], 'max_price': 500}
        self.assertEqual(judge.grade(product(), spec), 0)
        self.assertTrue(judge.violates(product(), spec))

    def test_empty_case(self):
        case = {'kind': 'empty'}
        self.assertEqual(judge.score_case(case, {'results': [], 'recommendations': []}, {})[0], 1.0)
        self.assertEqual(judge.score_case(case, {'results': [{'id': '1'}], 'recommendations': []},
                                          {'1': product()})[0], 0.0)


if __name__ == '__main__':
    unittest.main()
