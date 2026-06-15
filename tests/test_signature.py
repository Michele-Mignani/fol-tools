"""
tests/test_signature.py
=======================

Unit tests for FOLSignature.

Covers:
  - extract: relations, variables, constants, equality exclusion
  - classify: uppercase → const, digit-starting → const, bound lowercase → var,
    free lowercase → const
  - merge: union of two signatures, arity conflict detection
  - empty: blank signature
"""

import pytest
from fol_tools.parser import FOLParser
from fol_tools.signature import FOLSignature


def sig(formula: str) -> dict:
    tree = FOLParser().parse(formula)
    return FOLSignature().extract(tree)


class TestExtract:
    def test_unary_relation(self):
        s = sig("∀x Human(x)")
        assert s['Rel'] == {'Human': 1}
        assert 'x' in s['Var']

    def test_binary_relation(self):
        s = sig("∀x ∀y Loves(x, y)")
        assert s['Rel']['Loves'] == 2

    def test_ternary_relation(self):
        s = sig("∀x ∀y ∀z Between(x, y, z)")
        assert s['Rel']['Between'] == 3

    def test_zero_arity_relation(self):
        s = sig("∀x (Rain → Wet(x))")
        assert s['Rel']['Rain'] == 0
        assert s['Rel']['Wet'] == 1

    def test_uppercase_constant(self):
        s = sig("∀x Loves(x, John)")
        assert 'John' in s['Const']

    def test_numeric_constant(self):
        s = sig("∀x Sides(x, 3)")
        assert '3' in s['Const']
        assert s['Rel']['Sides'] == 2

    def test_bound_variable(self):
        s = sig("∀x Human(x)")
        assert 'x' in s['Var']

    def test_free_lowercase_treated_as_const(self):
        # pure lowercase, not bound → goes to Const
        s = sig("Loves(x, john)")
        assert 'john' in s['Const']

    def test_camelcase_treated_as_const(self):
        # camelCase (interior uppercase) → Const
        s = sig("In(diamondMine, wWE)")
        assert 'diamondMine' in s['Const']
        assert 'wWE' in s['Const']

    def test_multichar_lowercase_treated_as_const(self):
        # multi-char all-lowercase (tom, joey, istanbul) → Const, not Var
        s = sig("∃x (Owns(tom, x) ∧ WildTurkey(joey))")
        assert 'tom' in s['Const']
        assert 'joey' in s['Const']
        assert 'x' in s['Var']

    def test_var_pattern_single_letter_with_digit(self):
        # x1, y2 match [a-z][0-9]* → variable candidates when bound
        s = sig("∀x1 ∀y2 Loves(x1, y2)")
        assert 'x1' in s['Var']
        assert 'y2' in s['Var']

    def test_equality_in_rel(self):
        # = is now treated as a predicate symbol with arity 2
        s = sig("∀x ∀y (x = y → A(x))")
        assert '=' in s['Rel']
        assert s['Rel']['='] == 2

    def test_bool_const_contributes_nothing(self):
        s = sig("∀x (A(x) ∧ ⊤)")
        assert 'True' not in s['Rel']
        assert '⊤' not in s['Rel']

    def test_multiple_relations(self):
        s = sig("∀x (Human(x) ∧ Mortal(x))")
        assert set(s['Rel'].keys()) == {'Human', 'Mortal'}


class TestMerge:
    def test_disjoint_merge(self):
        s1 = sig("∀x Human(x)")
        s2 = sig("∀y Mortal(y)")
        merged = FOLSignature().merge(s1, s2)
        assert 'Human' in merged['Rel']
        assert 'Mortal' in merged['Rel']
        assert 'x' in merged['Var']
        assert 'y' in merged['Var']

    def test_same_arity_no_conflict(self):
        s1 = sig("∀x Human(x)")
        s2 = sig("∀y Human(y)")
        merged = FOLSignature().merge(s1, s2)
        assert merged['Rel']['Human'] == 1

    def test_arity_conflict_raises(self):
        s1 = {'Rel': {'R': 1}, 'Var': set(), 'Const': set()}
        s2 = {'Rel': {'R': 2}, 'Var': set(), 'Const': set()}
        with pytest.raises(ValueError, match="Arity conflict"):
            FOLSignature().merge(s1, s2)

    def test_const_union(self):
        s1 = sig("∀x Loves(x, John)")
        s2 = sig("∀y Loves(y, Mary)")
        merged = FOLSignature().merge(s1, s2)
        assert 'John' in merged['Const']
        assert 'Mary' in merged['Const']


class TestEmpty:
    def test_empty_signature(self):
        e = FOLSignature.empty()
        assert e == {'Rel': {}, 'Var': set(), 'Const': set()}
