"""Contract layer: published schemas, argument validation, and the Catalog facade used by server.py."""
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from engine.engine import SearchEngine

ROOT = Path(__file__).resolve().parent
INPUT_SCHEMA = json.loads((ROOT / 'schemas/input.schema.json').read_text())
OUTPUT_SCHEMA = json.loads((ROOT / 'schemas/output.schema.json').read_text())
VALIDATOR = Draft202012Validator(INPUT_SCHEMA)
STRING_FILTERS = ('category', 'product_type', 'color', 'audience', 'section', 'pattern')


def validate_arguments(arguments):
    # Reject non-JSON floating point values even when called directly from Python.
    json.dumps(arguments, allow_nan=False)
    try:
        VALIDATOR.validate(arguments)
    except ValidationError as exc:
        raise ValueError(exc.message) from exc
    filters = arguments.get('filters', {})
    if any(not filters[k].strip() for k in STRING_FILTERS if k in filters):
        raise ValueError('String filters cannot be blank')
    if filters.get('min_price', 0) > filters.get('max_price', math.inf):
        raise ValueError('min_price cannot exceed max_price')


class Catalog:
    """Contract boundary: validates arguments, then delegates to the hybrid search engine."""

    def __init__(self, products, dense='auto'):
        self.engine = SearchEngine(products, dense=dense)
        self.products = self.engine.products

    @classmethod
    def load(cls, path, dense='auto'):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')), dense=dense)

    def search(self, arguments):
        validate_arguments(arguments)
        return self.engine.search(arguments)

    def explain(self, arguments):
        """Same search plus the query plan, score breakdown and product cards (Search Lab UI)."""
        validate_arguments(arguments)
        return self.engine.run(arguments, explain=True)
