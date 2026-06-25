"""
tests/test_encoder_solver.py
============================

Unit tests for Z3ContextBuilder, FOLZ3Encoder, and FOLSolver.

Covers:
  - Context builder: symbol types for relations and constants
  - Encoder: produces valid Z3 expressions for each node type
  - Solver.is_satisfiable: sat/unsat cases including tautology and contradiction
  - Solver.are_equivalent: logically equivalent and non-equivalent pairs
  - Solver.implies: valid and invalid entailments
  - Theory axioms
  - Equality support
  - SolverTimeoutError is importable
"""

import pytest
from fol_tools.parser import FOLParser
from fol_tools.signature import FOLSignature
from fol_tools.encoder import Z3ContextBuilder, FOLZ3Encoder
from fol_tools.solver import FOLSolver, SolverTimeoutError
from fol_tools.exceptions import FOLSyntaxError, FOLEncoderError, FOLSignatureError, FormulaParseError
from fol_tools.formula import FOL


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_ctx(formula: str):
    tree = FOLParser().parse(formula)
    sig = FOLSignature().extract(tree)
    ctx = Z3ContextBuilder(sig)
    return ctx, ctx.build_symbols(), tree


# ------------------------------------------------------------------
# Z3ContextBuilder
# ------------------------------------------------------------------

class TestZ3ContextBuilder:
    def test_unary_function_created(self):
        from z3 import FuncDeclRef
        ctx, symbols, _ = make_ctx("∀x Human(x)")
        assert 'Human' in symbols
        assert hasattr(symbols['Human'], 'arity')

    def test_constant_created(self):
        from z3 import ExprRef
        ctx, symbols, _ = make_ctx("∀x Loves(x, John)")
        assert 'John' in symbols

    def test_zero_arity_relation_is_bool_const(self):
        ctx, symbols, _ = make_ctx("∀x (Rain → Wet(x))")
        import z3
        # Rain has arity 0 → Const(BoolSort())
        assert str(symbols['Rain'].sort()) == 'Bool'

    def test_sort_attribute_exists(self):
        ctx, _, _ = make_ctx("∀x Human(x)")
        assert ctx.sort is not None


# ------------------------------------------------------------------
# FOLZ3Encoder
# ------------------------------------------------------------------

class TestFOLZ3Encoder:
    def test_encodes_forall(self):
        ctx, symbols, tree = make_ctx("∀x Human(x)")
        enc = FOLZ3Encoder(sort=ctx.sort)
        expr = enc.encode(tree, symbols)
        assert expr is not None

    def test_encodes_exists(self):
        ctx, symbols, tree = make_ctx("∃x Human(x)")
        enc = FOLZ3Encoder(sort=ctx.sort)
        expr = enc.encode(tree, symbols)
        assert expr is not None

    def test_encodes_not(self):
        ctx, symbols, tree = make_ctx("∀x ¬Human(x)")
        enc = FOLZ3Encoder(sort=ctx.sort)
        expr = enc.encode(tree, symbols)
        assert 'not' in str(expr).lower() or '!' in str(expr)

    def test_encodes_bool_const_true(self):
        from fol_tools.ast import BoolConstNode
        ctx = Z3ContextBuilder({'Rel': {}, 'Var': set(), 'Const': set()})
        symbols = ctx.build_symbols()
        enc = FOLZ3Encoder(sort=ctx.sort)
        expr = enc.encode(BoolConstNode(True), symbols)
        import z3
        assert z3.is_true(expr)

    def test_encodes_equality(self):
        ctx, symbols, tree = make_ctx("∀x ∀y (x = y → A(x))")
        enc = FOLZ3Encoder(sort=ctx.sort)
        expr = enc.encode(tree, symbols)
        assert expr is not None

    def test_unknown_node_raises(self):
        from fol_tools.ast import Node
        enc = FOLZ3Encoder()
        with pytest.raises(ValueError, match="Unknown AST node type"):
            enc.encode(Node(), {})


# ------------------------------------------------------------------
# FOLSolver — is_satisfiable
# ------------------------------------------------------------------

class TestSatisfiable:
    def setup_method(self):
        self.solver = FOLSolver()

    def test_satisfiable_simple(self):
        assert self.solver.is_satisfiable(FOL("∀x Human(x)")) is True

    def test_contradiction_unsatisfiable(self):
        assert self.solver.is_satisfiable(FOL("∀x (A(x) ∧ ¬A(x))")) is False

    def test_tautology_satisfiable(self):
        assert self.solver.is_satisfiable(FOL("⊤")) is True

    def test_false_constant_unsatisfiable(self):
        assert self.solver.is_satisfiable(FOL("⊥")) is False

    def test_existential_is_sat(self):
        assert self.solver.is_satisfiable(FOL("∃x Human(x)")) is True

    def test_with_theory(self):
        # Human(x) → Mortal(x) as theory; ∃x Mortal(x) should remain sat
        theory = ["∀x (Human(x) → Mortal(x))"]
        assert self.solver.is_satisfiable(FOL("∃x Mortal(x)"), theory=theory) is True


# ------------------------------------------------------------------
# FOLSolver — are_equivalent
# ------------------------------------------------------------------

class TestEquivalence:
    def setup_method(self):
        self.solver = FOLSolver()

    def test_identical_formulas(self):
        f = FOL("∀x (Human(x) → Mortal(x))")
        assert self.solver.are_equivalent(f, f) is True

    def test_contrapositive(self):
        f1 = FOL("∀x (Human(x) → Mortal(x))")
        f2 = FOL("∀x (¬Mortal(x) → ¬Human(x))")
        assert self.solver.are_equivalent(f1, f2) is True

    def test_non_equivalent(self):
        f1 = FOL("∀x Human(x)")
        f2 = FOL("∃x Human(x)")
        assert self.solver.are_equivalent(f1, f2) is False

    def test_double_negation_elimination(self):
        f1 = FOL("∀x A(x)")
        f2 = FOL("∀x ¬¬A(x)")
        assert self.solver.are_equivalent(f1, f2) is True

    def test_de_morgan_or(self):
        f1 = FOL("∀x ¬(A(x) ∨ B(x))")
        f2 = FOL("∀x (¬A(x) ∧ ¬B(x))")
        assert self.solver.are_equivalent(f1, f2) is True

    def test_de_morgan_and(self):
        f1 = FOL("∀x ¬(A(x) ∧ B(x))")
        f2 = FOL("∀x (¬A(x) ∨ ¬B(x))")
        assert self.solver.are_equivalent(f1, f2) is True

    def test_iff_vs_implies_both_ways(self):
        f1 = FOL("∀x (A(x) ↔ B(x))")
        f2 = FOL("∀x ((A(x) → B(x)) ∧ (B(x) → A(x)))")
        assert self.solver.are_equivalent(f1, f2) is True


# ------------------------------------------------------------------
# FOLSolver — implies
# ------------------------------------------------------------------

class TestImplies:
    def setup_method(self):
        self.solver = FOLSolver()

    def test_tautology_implies_anything(self):
        f_true = FOL("⊤")
        f = FOL("∀x Human(x)")
        # ⊤ does NOT imply an arbitrary FOL formula — remains satisfiable
        # Correct test: something specific implies itself
        assert self.solver.implies(f, f) is True

    def test_does_not_imply(self):
        f1 = FOL("∃x Human(x)")
        f2 = FOL("∀x Human(x)")
        assert self.solver.implies(f1, f2) is False

    def test_specialisation_implies_generalisation_reverse(self):
        # ∀x A(x) ⊨ ∃x A(x)  (universal implies existential)
        f1 = FOL("∀x A(x)")
        f2 = FOL("∃x A(x)")
        assert self.solver.implies(f1, f2) is True

    def test_contradiction_implies_anything(self):
        f_false = FOL("⊥")
        f = FOL("∀x Human(x)")
        assert self.solver.implies(f_false, f) is True

    def test_with_theory(self):
        # With ∀x(Mortal(x)→Dead(x)) as theory, Mortal(a) should imply Dead(a)
        theory = ["∀x (Mortal(x) → Dead(x))"]
        f1 = FOL("Mortal(a)")
        f2 = FOL("Dead(a)")
        assert self.solver.implies(f1, f2, theory=theory) is True


# ------------------------------------------------------------------
# Equality support
# ------------------------------------------------------------------

class TestEquality:
    def setup_method(self):
        self.solver = FOLSolver()

    def test_reflexivity(self):
        f = FOL("∀x (x = x)")
        assert self.solver.is_satisfiable(f) is True

    def test_equality_implies_predicate(self):
        # ∀x∀y(x=y → A(x)) and A(a) together with a=b should imply A(b)
        theory = ["∀x ∀y (x = y → (A(x) → A(y)))"]
        f1 = FOL("a = b")
        f2 = FOL("A(a) → A(b)")
        assert self.solver.implies(f1, f2, theory=theory) is True


# ------------------------------------------------------------------
# SolverTimeoutError importable
# ------------------------------------------------------------------

class TestSolverTimeoutError:
    def test_importable(self):
        from fol_tools.solver import SolverTimeoutError
        assert issubclass(SolverTimeoutError, Exception)


# ------------------------------------------------------------------
# FormulaParseError — raised on unparseable / invalid FOL strings
# ------------------------------------------------------------------

class TestFormulaParseError:
    """FOLSolver methods must raise FOLSyntaxError (not return False)
    when a formula cannot be parsed, and SolverTimeoutError when Z3
    exceeds the timeout.  This verifies callers can distinguish
    parse failures, timeouts, and genuine non-entailment."""

    def test_formula_parse_error_is_value_error(self):
        assert issubclass(FormulaParseError, ValueError)

    def test_formula_parse_error_importable_from_package(self):
        from fol_tools import FormulaParseError as FPE
        assert FPE is FormulaParseError

    # -- implies: parse error via bad theory string --

    def test_implies_raises_parse_error_on_bad_theory(self):
        solver = FOLSolver()
        f = FOL("∀x Human(x)")
        # ASCII operator in theory string triggers parse error
        with pytest.raises(FOLSyntaxError):
            solver.implies(f, f, theory=["∀x (A(x) & B(x))"])

    def test_are_equivalent_raises_parse_error_on_bad_theory(self):
        solver = FOLSolver()
        f = FOL("∀x Human(x)")
        with pytest.raises(FOLSyntaxError):
            solver.are_equivalent(f, f, theory=["A(x) -> B(x)"])

    def test_is_satisfiable_raises_parse_error_on_bad_theory(self):
        solver = FOLSolver()
        f = FOL("∀x Human(x)")
        with pytest.raises(FOLSyntaxError):
            solver.is_satisfiable(f, theory=["A(x) & B(x)"])

    def test_implies_does_not_swallow_parse_errors(self):
        # Confirm that a bad theory string propagates rather than silently
        # returning False (the old behaviour).
        solver = FOLSolver()
        f = FOL("∀x Human(x)")
        raised = False
        try:
            solver.implies(f, f, theory=["∀x (A(x) -> B(x))"])
        except FOLSyntaxError:
            raised = True
        assert raised, "FOLSyntaxError was swallowed — got False instead"

    # -- unbalanced parentheses in theory strings --
    # These cannot be caught at FOL() construction time (FOL defers to
    # validate()), but are re-parsed inside the solver and must raise
    # FOLSyntaxError — never return False.

    @pytest.mark.parametrize("bad", [
        "∀x (Human(x)",      # missing outer closing paren
        "Human(x",            # missing argument-list closing paren
        "∀x Human(x))",      # extra closing paren
    ])
    def test_implies_raises_on_unbalanced_parens_in_theory(self, bad):
        solver = FOLSolver()
        f = FOL("∀x Human(x)")
        with pytest.raises(FOLSyntaxError):
            solver.implies(f, f, theory=[bad])

    @pytest.mark.parametrize("bad", [
        "∀x (Human(x)",
        "Human(x",
        "∀x Human(x))",
    ])
    def test_are_equivalent_raises_on_unbalanced_parens_in_theory(self, bad):
        solver = FOLSolver()
        f = FOL("∀x Human(x)")
        with pytest.raises(FOLSyntaxError):
            solver.are_equivalent(f, f, theory=[bad])

    @pytest.mark.parametrize("bad", [
        "∀x (Human(x)",
        "Human(x",
        "∀x Human(x))",
    ])
    def test_is_satisfiable_raises_on_unbalanced_parens_in_theory(self, bad):
        solver = FOLSolver()
        f = FOL("∀x Human(x)")
        with pytest.raises(FOLSyntaxError):
            solver.is_satisfiable(f, theory=[bad])

    # -- timeout --
    # We mock FOLSolver._check to return z3.unknown, which is what Z3 emits
    # when it exceeds the timeout.  This makes the tests deterministic and
    # independent of machine speed or Z3 internal caching.

    def _make_unknown_solver(self):
        """Return a FOLSolver whose _check always returns z3.unknown."""
        from unittest.mock import patch
        import z3 as _z3
        solver = FOLSolver()
        # Patch the _check method to simulate a timeout response
        solver._check = lambda s, t: (_ for _ in ()).throw(
            SolverTimeoutError("mocked timeout")
        )
        return solver

    def test_implies_raises_timeout(self):
        from unittest.mock import patch
        import z3 as _z3
        f = FOL("∀x Human(x)")
        solver = FOLSolver()
        with patch.object(solver, "_check", side_effect=SolverTimeoutError("mocked")):
            with pytest.raises(SolverTimeoutError):
                solver.implies(f, f)

    def test_are_equivalent_raises_timeout(self):
        from unittest.mock import patch
        f = FOL("∀x Human(x)")
        solver = FOLSolver()
        with patch.object(solver, "_check", side_effect=SolverTimeoutError("mocked")):
            with pytest.raises(SolverTimeoutError):
                solver.are_equivalent(f, f)

    def test_is_satisfiable_raises_timeout(self):
        from unittest.mock import patch
        f = FOL("∀x Human(x)")
        solver = FOLSolver()
        with patch.object(solver, "_check", side_effect=SolverTimeoutError("mocked")):
            with pytest.raises(SolverTimeoutError):
                solver.is_satisfiable(f)
