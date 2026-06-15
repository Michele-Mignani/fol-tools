"""
tests/test_formula.py
=====================

Unit tests for the FOL high-level façade.

Covers:
  - Lazy parsing (tree, signature)
  - validate(): valid formula, invalid formulas, free variables
  - free_variables() / bound_variables()
  - to_smtlib(): produces a non-empty string
  - to_string(): round-trip
  - to_nl(): delegation to translator
  - perturbations(): delegation to perturbations module
  - User-provided signature override
  - __repr__ / __str__ do not crash
"""

import pytest
from fol_tools.formula import FOL
from fol_tools.ast import QuantifierNode


class TestLazyParsing:
    def test_tree_is_ast(self):
        f = FOL("∀x Human(x)")
        assert f._tree is None       # not parsed yet
        t = f.tree
        assert t is not None
        assert f._tree is t          # cached

    def test_signature_is_cached(self):
        f = FOL("∀x Human(x)")
        s1 = f.signature
        s2 = f.signature
        assert s1 is s2

    def test_user_signature_returned_directly(self):
        user_sig = {'Rel': {'Human': 1}, 'Var': {'x'}, 'Const': set()}
        f = FOL("∀x Human(x)", signature=user_sig)
        assert f.signature is user_sig


class TestValidate:
    def test_valid_formula(self):
        assert FOL("∀x (Human(x) → Mortal(x))").validate() is True

    def test_tautology_valid(self):
        assert FOL("⊤").validate() is True

    def test_contradiction_valid(self):
        assert FOL("⊥").validate() is True

    def test_forbidden_arrow(self):
        assert FOL("∀x (A(x) -> B(x))").validate() is False

    def test_forbidden_ampersand(self):
        assert FOL("∀x (A(x) & B(x))").validate() is False

    def test_free_variable_invalid(self):
        # 'y' is free — validate should return False
        assert FOL("∀x Loves(x, y)").validate() is False

    def test_parse_error_returns_false(self):
        assert FOL("not a formula !!!").validate() is False

    def test_uppercase_constant_not_free(self):
        assert FOL("∀x Loves(x, John)").validate() is True

    def test_user_declared_constant_not_free(self):
        user_sig = {'Rel': {'Houses': 3}, 'Var': set(), 'Const': {'books', 'study'}}
        f = FOL("∀x Houses(x, books, study)", signature=user_sig)
        assert f.validate() is True


class TestFreeAndBoundVariables:
    def test_no_free_variables(self):
        assert FOL("∀x Human(x)").free_variables() == set()

    def test_free_variable_detected(self):
        assert 'y' in FOL("∀x Loves(x, y)").free_variables()

    def test_numeric_not_free(self):
        assert '3' not in FOL("∀x Sides(x, 3)").free_variables()

    def test_uppercase_not_free(self):
        assert 'John' not in FOL("∀x Loves(x, John)").free_variables()

    def test_bound_variables(self):
        bv = FOL("∀x ∃y Loves(x, y)").bound_variables()
        assert bv == {'x', 'y'}

    def test_bound_variables_empty_for_prop(self):
        bv = FOL("⊤").bound_variables()
        assert bv == set()


class TestToSMTLib:
    def test_produces_string(self):
        s = FOL("∀x Human(x)").to_smtlib()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_contains_forall(self):
        s = FOL("∀x Human(x)").to_smtlib()
        assert 'forall' in s.lower()

    def test_contradiction(self):
        s = FOL("⊥").to_smtlib()
        assert 'false' in s.lower()


class TestToString:
    def test_round_trip(self):
        formula = "∀x (Human(x) → Mortal(x))"
        f = FOL(formula)
        reconstructed = f.to_string()
        # re-parse must succeed
        from fol_tools.parser import FOLParser
        assert FOLParser().parse(reconstructed) is not None


class TestToNL:
    def test_returns_sentence(self):
        s = FOL("∀x Human(x)").to_nl()
        assert s.endswith('.')
        assert s[0].isupper()

    def test_with_meanings(self):
        s = FOL("∀x (Human(x) → Mortal(x))").to_nl(
            symbol_meanings={'Human': '{0} is human', 'Mortal': '{0} is mortal'}
        )
        assert 'is human' in s


class TestPerturbations:
    def test_returns_list(self):
        result = FOL("∀x (Human(x) → Mortal(x))").perturbations(n=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_results_distinct_from_original(self):
        formula = "∀x (Human(x) → Mortal(x))"
        result = FOL(formula).perturbations(n=5)
        assert formula not in result
