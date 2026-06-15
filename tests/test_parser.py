"""
tests/test_parser.py
====================

Unit tests for FOLParser.

Covers:
  1. Tokeniser — identifiers, numerics, operators, separators
  2. Operator precedence (↔ < ⊕ < → < ∨ < ∧ < ¬)
  3. Quantifier scope (limited-scope rule)
  4. All atom forms: relation, equality, inequality, propositional, bool const
  5. Numeric constants as relation arguments
  6. Nested quantifiers
  7. Error cases: forbidden ASCII, lowercase relation, empty formula,
     unbalanced parens, unexpected characters, bad quantifier variable
"""

import pytest
from fol_tools.parser import FOLParser
from fol_tools.ast import (
    QuantifierNode, BooleanNode, RelationNode, BoolConstNode,
)


def parse(s: str):
    return FOLParser().parse(s)


# ============================================================
# 1. Basic atoms
# ============================================================

class TestAtoms:
    def test_unary_relation(self):
        t = parse("∀x Human(x)")
        assert isinstance(t, QuantifierNode)
        assert isinstance(t.body, RelationNode)
        assert t.body.name == 'Human'
        assert t.body.arguments == ['x']

    def test_binary_relation(self):
        t = parse("∀x ∀y Loves(x, y)")
        inner = t.body
        assert isinstance(inner, QuantifierNode)
        assert inner.body.name == 'Loves'
        assert inner.body.arguments == ['x', 'y']

    def test_zero_arity_relation(self):
        t = parse("∀x (Rain → Wet(x))")
        assert t is not None

    def test_bool_true(self):
        t = parse("⊤")
        assert isinstance(t, BoolConstNode) and t.value is True

    def test_bool_false(self):
        t = parse("⊥")
        assert isinstance(t, BoolConstNode) and t.value is False

    def test_bool_True_keyword(self):
        t = parse("True")
        assert isinstance(t, BoolConstNode) and t.value is True

    def test_bool_False_keyword(self):
        t = parse("False")
        assert isinstance(t, BoolConstNode) and t.value is False

    def test_equality(self):
        t = parse("∀x ∀y (x = y → A(x))")
        assert t is not None

    def test_inequality_desugars_to_negated_equality(self):
        t = parse("∀x ∀y (x ≠ y → A(x))")
        # body of ∀x is ∀y; body of ∀y is implies
        impl = t.body.body
        assert impl.operator == 'implies'
        lhs = impl.children[0]
        assert isinstance(lhs, BooleanNode)
        assert lhs.operator == 'not'
        assert isinstance(lhs.children[0], RelationNode)
        assert lhs.children[0].name == '='


# ============================================================
# 2. Operator precedence
# ============================================================

class TestPrecedence:
    def test_implication_binds_looser_than_disjunction(self):
        # A → B ∨ C  should parse as  A → (B ∨ C)
        t = parse("∀x (A(x) → B(x) ∨ C(x))")
        impl = t.body
        assert impl.operator == 'implies'
        assert impl.children[1].operator == 'or'

    def test_conjunction_binds_tighter_than_disjunction(self):
        # A ∨ B ∧ C  should parse as  A ∨ (B ∧ C)
        t = parse("∀x (A(x) ∨ B(x) ∧ C(x))")
        assert t.body.operator == 'or'
        assert t.body.children[1].operator == 'and'

    def test_negation_tighter_than_conjunction(self):
        # ¬A ∧ B  should parse as  (¬A) ∧ B
        t = parse("∀x (¬A(x) ∧ B(x))")
        assert t.body.operator == 'and'
        assert t.body.children[0].operator == 'not'

    def test_implication_right_associative(self):
        # A → B → C  should parse as  A → (B → C)
        t = parse("∀x (A(x) → B(x) → C(x))")
        impl = t.body
        assert impl.operator == 'implies'
        assert impl.children[1].operator == 'implies'

    def test_biconditional_left_associative(self):
        # A ↔ B ↔ C  should parse as  (A ↔ B) ↔ C
        t = parse("∀x (A(x) ↔ B(x) ↔ C(x))")
        outer = t.body
        assert outer.operator == 'iff'
        assert outer.children[0].operator == 'iff'

    def test_biconditional_looser_than_implication(self):
        # A → B ↔ C → D  parses as  (A → B) ↔ (C → D)
        t = parse("∀x (A(x) → B(x) ↔ C(x) → D(x))")
        assert t.body.operator == 'iff'
        assert t.body.children[0].operator == 'implies'

    def test_xor_between_impl_and_iff(self):
        # A → B ⊕ C ↔ D  is evaluated as  (A → B) ⊕ C … no: ⊕ > → > ↔
        # so A → (B ⊕ C) ↔ D is wrong; correct: (A → B) is lowest…
        # let's just check xor binds tighter than iff
        t = parse("∀x (A(x) ⊕ B(x) ↔ C(x))")
        assert t.body.operator == 'iff'
        assert t.body.children[0].operator == 'xor'


# ============================================================
# 3. Quantifier scope
# ============================================================

class TestQuantifierScope:
    def test_limited_scope_before_implication(self):
        # ∃y(P(y)) → Q(y)  parses as  (∃y P(y)) → Q(y)
        # The implication should be at the top level, not inside the quantifier
        t = parse("∃y P(y) → Q(y)")
        # t is an implication where the left is the quantifier
        assert isinstance(t, BooleanNode)
        assert t.operator == 'implies'
        assert isinstance(t.children[0], QuantifierNode)

    def test_chained_quantifiers(self):
        t = parse("∀x ∃y Loves(x, y)")
        assert isinstance(t, QuantifierNode)
        assert t.variable == 'x'
        assert isinstance(t.body, QuantifierNode)
        assert t.body.variable == 'y'

    def test_quantifier_with_negation(self):
        t = parse("∀x ¬A(x)")
        assert isinstance(t, QuantifierNode)
        assert t.body.operator == 'not'

    def test_nested_quantifier_with_parens(self):
        t = parse("∀x (∃y Loves(x, y))")
        assert t.body.quantifier == 'exists'


# ============================================================
# 4. Numeric constants
# ============================================================

class TestNumericConstants:
    def test_single_digit(self):
        t = parse("∀x Sides(x, 3)")
        assert t.body.arguments == ['x', '3']

    def test_multi_digit(self):
        t = parse("∀x HasAge(x, 42)")
        assert t.body.arguments == ['x', '42']

    def test_two_numerics(self):
        t = parse("∀x Between(x, 1, 2)")
        assert t.body.arguments == ['x', '1', '2']

    def test_malls_regression(self):
        # regression: ∀x (Polygon(x) ∧ Sides(x, 3) ∧ Angles(x, 3) → Triangle(x))
        t = parse("∀x (Polygon(x) ∧ Sides(x, 3) ∧ Angles(x, 3) → Triangle(x))")
        assert t is not None


# ============================================================
# 5. Error cases
# ============================================================

class TestErrors:
    def test_empty_formula(self):
        with pytest.raises(ValueError):
            parse("")

    def test_forbidden_arrow(self):
        with pytest.raises(ValueError, match="Forbidden"):
            parse("∀x (A(x) -> B(x))")

    def test_forbidden_ampersand(self):
        with pytest.raises(ValueError, match="Forbidden"):
            parse("∀x (A(x) & B(x))")

    def test_unbalanced_open_paren(self):
        with pytest.raises(Exception):
            parse("∀x (A(x) → B(x)")

    def test_unbalanced_close_paren(self):
        with pytest.raises(Exception):
            parse("∀x A(x) → B(x))")

    def test_unexpected_character(self):
        with pytest.raises(ValueError, match="Unexpected character"):
            parse("∀x A@(x)")

    def test_lowercase_relation_name(self):
        with pytest.raises(ValueError):
            parse("∀x human(x)")

    def test_uppercase_quantifier_variable(self):
        with pytest.raises(ValueError):
            parse("∀X A(X)")

    def test_numeric_token_standalone(self):
        with pytest.raises(ValueError):
            parse("42")

    def test_trailing_token(self):
        with pytest.raises(ValueError):
            parse("∀x A(x) garbage")
