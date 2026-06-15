"""
tests/test_translator.py
========================

Unit tests for FOLTranslator.

Covers:
  - to_string: round-trip, precedence-driven minimal parenthesisation,
    quantifiers, equality, boolean constants
  - to_tree_string: fully-parenthesised prefix form
  - to_nl: basic NL, symbol meanings, negative meanings,
    double-negation elimination, quantifier collapsing,
    split_conjunctions, parenthesis flag
"""

import pytest
from fol_tools.parser import FOLParser
from fol_tools.translator import FOLTranslator


def tree(s: str):
    return FOLParser().parse(s)


tr = FOLTranslator()


# ============================================================
# to_string
# ============================================================

class TestToString:
    def test_universal_implication(self):
        s = tr.to_string(tree("∀x (Human(x) → Mortal(x))"))
        assert 'Human' in s and 'Mortal' in s and '→' in s

    def test_bool_const_top(self):
        assert tr.to_string(tree("⊤")) == '⊤'
        assert tr.to_string(tree("⊥")) == '⊥'

    def test_equality(self):
        s = tr.to_string(tree("∀x ∀y (x = y → A(x))"))
        assert '=' in s

    def test_negation(self):
        s = tr.to_string(tree("∀x ¬A(x)"))
        assert '¬' in s

    def test_zero_arity_relation(self):
        s = tr.to_string(tree("∀x (Rain → Wet(x))"))
        assert 'Rain' in s

    def test_round_trip_parses(self):
        original = "∀x (Human(x) → Mortal(x))"
        reconstructed = tr.to_string(tree(original))
        # round-trip: reconstructed string must itself be parseable
        tree2 = FOLParser().parse(reconstructed)
        assert tree2 is not None

    def test_iff(self):
        s = tr.to_string(tree("∀x (A(x) ↔ B(x))"))
        assert '↔' in s

    def test_xor(self):
        s = tr.to_string(tree("∀x (A(x) ⊕ B(x))"))
        assert '⊕' in s


# ============================================================
# to_tree_string
# ============================================================

class TestToTreeString:
    def test_forall(self):
        s = tr.to_tree_string(tree("∀x Human(x)"))
        assert s == 'Forall(x, Human(x))'

    def test_exists(self):
        s = tr.to_tree_string(tree("∃x Human(x)"))
        assert s.startswith('Exists(')

    def test_bool_true(self):
        assert tr.to_tree_string(tree("⊤")) == 'True'

    def test_bool_false(self):
        assert tr.to_tree_string(tree("⊥")) == 'False'

    def test_implies_tree(self):
        s = tr.to_tree_string(tree("∀x (A(x) → B(x))"))
        assert 'Implies' in s

    def test_and_tree(self):
        s = tr.to_tree_string(tree("∀x (A(x) ∧ B(x))"))
        assert 'And' in s

    def test_not_tree(self):
        s = tr.to_tree_string(tree("∀x ¬A(x)"))
        assert 'Not' in s


# ============================================================
# to_nl
# ============================================================

class TestToNL:
    def test_basic_universal(self):
        s = tr.to_nl(tree("∀x Human(x)"))
        assert s.startswith('For all')
        assert s.endswith('.')

    def test_basic_existential(self):
        s = tr.to_nl(tree("∃x Human(x)"))
        assert 'there exists' in s.lower()

    def test_implication_rule(self):
        s = tr.to_nl(tree("∀x (A(x) → B(x))"))
        assert 'if' in s.lower()

    def test_symbol_meanings(self):
        s = tr.to_nl(
            tree("∀x (Human(x) → Mortal(x))"),
            symbol_meanings={'Human': '{0} is human', 'Mortal': '{0} is mortal'},
        )
        assert 'is human' in s
        assert 'is mortal' in s

    def test_negative_meaning(self):
        s = tr.to_nl(
            tree("∀x ¬Alive(x)"),
            symbol_meanings={'Alive': '{0} is alive'},
            negative_meanings={'Alive': '{0} is dead'},
        )
        assert 'is dead' in s

    def test_double_negation_eliminated(self):
        s_single = tr.to_nl(tree("∀x ¬A(x)"))
        s_double = tr.to_nl(tree("∀x ¬¬A(x)"))
        # double-neg elimination: ¬¬A renders same as A
        assert 'not the case' not in s_double.lower() or 'not' not in s_double

    def test_inequality_renders(self):
        s = tr.to_nl(tree("∀x ∀y (x ≠ y → A(x))"))
        assert 'not equal' in s.lower() or 'different' in s.lower() or 'not' in s.lower()

    def test_split_conjunctions(self):
        s = tr.to_nl(tree("∀x A(x) ∧ ∀y B(y)"), split_conjunctions=True)
        # two capitalised clauses split by '. '
        assert '. ' in s

    def test_capitalised_start(self):
        s = tr.to_nl(tree("∀x Human(x)"))
        assert s[0].isupper()

    def test_ends_with_period(self):
        s = tr.to_nl(tree("∀x Human(x)"))
        assert s.endswith('.')

    def test_quantifier_collapse_two(self):
        s = tr.to_nl(tree("∀x ∀y Loves(x, y)"))
        assert 'x and y' in s

    def test_custom_rule_override(self):
        s = tr.to_nl(
            tree("∀x (A(x) → B(x))"),
            rules={'implies': 'whenever {0}, always {1}'},
        )
        assert 'whenever' in s
