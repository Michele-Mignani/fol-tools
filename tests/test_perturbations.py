"""
tests/test_perturbations.py
===========================

Unit tests for the perturbations module.

Covers:
  - Each individual transform function
  - generate_rule_based: count, uniqueness, bad input gracefully handled
  - RULE_TRANSFORMS is a list of callables
"""

import pytest
from fol_tools.parser import FOLParser
from fol_tools.translator import FOLTranslator
from fol_tools.perturbations import (
    flip_root_quantifier,
    flip_root_connective,
    negate_whole,
    negate_consequent,
    swap_nested_quantifiers,
    generate_rule_based,
    RULE_TRANSFORMS,
)

_parser = FOLParser()
_tr = FOLTranslator()


def parse(s):
    return _parser.parse(s)


def to_str(node):
    return _tr.to_string(node)


# ============================================================
# Individual transforms
# ============================================================

class TestFlipRootQuantifier:
    def test_forall_becomes_exists(self):
        result = flip_root_quantifier(parse("∀x Human(x)"))
        assert result.quantifier == 'exists'

    def test_exists_becomes_forall(self):
        result = flip_root_quantifier(parse("∃x Human(x)"))
        assert result.quantifier == 'forall'

    def test_non_quantifier_returns_none(self):
        result = flip_root_quantifier(parse("A(x) ∧ B(x)"))
        assert result is None

    def test_does_not_mutate_input(self):
        t = parse("∀x Human(x)")
        original_q = t.quantifier
        flip_root_quantifier(t)
        assert t.quantifier == original_q


class TestFlipRootConnective:
    def test_implies_becomes_iff(self):
        result = flip_root_connective(parse("∀x (A(x) → B(x))"))
        assert result is not None
        assert result.body.operator == 'iff'

    def test_iff_becomes_implies(self):
        result = flip_root_connective(parse("∀x (A(x) ↔ B(x))"))
        assert result.body.operator == 'implies'

    def test_and_becomes_or(self):
        result = flip_root_connective(parse("∀x (A(x) ∧ B(x))"))
        assert result.body.operator == 'or'

    def test_or_becomes_and(self):
        result = flip_root_connective(parse("∀x (A(x) ∨ B(x))"))
        assert result.body.operator == 'and'

    def test_no_applicable_connective(self):
        result = flip_root_connective(parse("∀x ¬A(x)"))
        assert result is None


class TestNegateWhole:
    def test_wraps_in_not(self):
        result = negate_whole(parse("∀x Human(x)"))
        assert result.operator == 'not'

    def test_always_returns_node(self):
        result = negate_whole(parse("⊤"))
        assert result is not None


class TestNegateConsequent:
    def test_negates_consequent(self):
        result = negate_consequent(parse("∀x (Human(x) → Mortal(x))"))
        assert result is not None
        # body of ∀x is implies; right child is now 'not Mortal'
        impl = result.body
        assert impl.operator == 'implies'
        assert impl.children[1].operator == 'not'

    def test_through_quantifier(self):
        result = negate_consequent(parse("∀x ∃y (A(x) → B(y))"))
        assert result is not None

    def test_no_implication_returns_none(self):
        result = negate_consequent(parse("∀x (A(x) ∧ B(x))"))
        assert result is None


class TestSwapNestedQuantifiers:
    def test_forall_exists_becomes_exists_forall(self):
        result = swap_nested_quantifiers(parse("∀x ∃y Loves(x, y)"))
        assert result is not None
        assert result.quantifier == 'exists'
        assert result.variable == 'y'
        assert result.body.quantifier == 'forall'
        assert result.body.variable == 'x'

    def test_single_quantifier_returns_none(self):
        result = swap_nested_quantifiers(parse("∀x Human(x)"))
        assert result is None

    def test_no_quantifier_returns_none(self):
        result = swap_nested_quantifiers(parse("A(x) ∧ B(x)"))
        assert result is None


# ============================================================
# generate_rule_based
# ============================================================

class TestGenerateRuleBased:
    def test_returns_list(self):
        result = generate_rule_based("∀x (Human(x) → Mortal(x))", n=5)
        assert isinstance(result, list)

    def test_all_distinct(self):
        result = generate_rule_based("∀x (Human(x) → Mortal(x))", n=10)
        assert len(result) == len(set(result))

    def test_original_not_in_result(self):
        formula = "∀x (Human(x) → Mortal(x))"
        result = generate_rule_based(formula, n=10)
        assert formula not in result

    def test_respects_n_limit(self):
        result = generate_rule_based("∀x (Human(x) → Mortal(x))", n=2)
        assert len(result) <= 2

    def test_invalid_formula_returns_empty(self):
        result = generate_rule_based("this is not a formula !@#", n=5)
        assert result == []

    def test_results_are_parseable(self):
        results = generate_rule_based("∀x (Human(x) → Mortal(x))", n=5)
        for s in results:
            t = FOLParser().parse(s)
            assert t is not None

    def test_n_zero_returns_empty(self):
        result = generate_rule_based("∀x Human(x)", n=0)
        assert result == []


# ============================================================
# RULE_TRANSFORMS registry
# ============================================================

class TestRuleTransforms:
    def test_is_list(self):
        assert isinstance(RULE_TRANSFORMS, list)

    def test_all_callable(self):
        for fn in RULE_TRANSFORMS:
            assert callable(fn)

    def test_contains_known_transforms(self):
        names = {fn.__name__ for fn in RULE_TRANSFORMS}
        assert 'flip_root_quantifier' in names
        assert 'negate_whole' in names
