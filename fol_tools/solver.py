"""
fol_tools.solver
================

Z3-backed semantic reasoner for FOL formulas.

Provides three core operations on :class:`~fol_tools.formula.FOL` objects:

* **satisfiability** — does a model exist?
* **equivalence** — do two formulas hold in exactly the same models?
* **entailment** — does every model of F₁ also satisfy F₂?

All three operations accept an optional *theory*: a list of extra axiom
strings that are asserted before the main check.  This is useful when the
domain has background knowledge (e.g. transitivity of a relation).

Timeout
-------
Each call accepts a ``timeout`` parameter (milliseconds).  When Z3
exceeds the limit it returns the special result ``z3.unknown``; the
solver re-raises this as :exc:`SolverTimeoutError`.

Equality support
----------------
When any formula in a call uses the ``'='`` predicate, the solver
automatically adds explicit reflexivity, symmetry, and transitivity
axioms.  Z3's native ``==`` already satisfies these, so the axioms are
redundant but make the equality theory self-contained and visible in the
solver's assertion stack for debugging.

Usage
-----
.. code-block:: python

    from fol_tools.formula import FOL
    from fol_tools.solver import FOLSolver, SolverTimeoutError

    solver = FOLSolver()

    f1 = FOL("∀x (Human(x) → Mortal(x))")
    f2 = FOL("∀x (¬Mortal(x) → ¬Human(x))")

    solver.are_equivalent(f1, f2)   # True  (contrapositive)
    solver.implies(f1, f2)          # True

    solver.is_satisfiable(FOL("∀x (A(x) ∧ ¬A(x))"))  # False

Thread safety
-------------
:class:`FOLSolver` is stateless between calls.  Multiple threads can
share a single instance safely, as long as they do not share
:class:`~fol_tools.formula.FOL` objects (which cache lazy state).
"""

from __future__ import annotations

from z3 import And, ForAll, Implies, Not, Solver, sat, unsat

from .encoder import FOLZ3Encoder, Z3ContextBuilder
from .signature import FOLSignature


class SolverTimeoutError(Exception):
    """Raised when a Z3 solver call exceeds the specified timeout.

    Attributes
    ----------
    message : str
        Human-readable description including the timeout duration.
    """


class FOLSolver:
    """Semantic reasoner backed by the Z3 SMT solver.

    Parameters
    ----------
    default_timeout : int
        Default timeout in milliseconds applied to every call that does
        not override it explicitly.  Default is 10 000 ms (10 seconds).

    Methods
    -------
    is_satisfiable(formula, theory, timeout)
        Check whether *formula* has at least one model.
    are_equivalent(f1, f2, theory, timeout)
        Check whether *f1* and *f2* are logically equivalent.
    implies(f1, f2, theory, timeout)
        Check whether every model of *f1* also satisfies *f2*.
    """

    DEFAULT_TIMEOUT: int = 10_000   # milliseconds

    def __init__(self, default_timeout: int = DEFAULT_TIMEOUT) -> None:
        self._default_timeout = default_timeout

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def is_satisfiable(
        self,
        formula,
        theory: list[str] | None = None,
        timeout: int | None = None,
    ) -> bool:
        """Return ``True`` if *formula* is satisfiable.

        Parameters
        ----------
        formula : FOL
            The formula to check.
        theory : list[str], optional
            Extra axiom strings.  Each string is parsed and asserted
            before the main formula.
        timeout : int, optional
            Timeout in milliseconds.  Overrides the instance default.
            Pass ``None`` to use the instance default.

        Returns
        -------
        bool
            ``True`` if satisfiable, ``False`` if unsatisfiable or
            malformed (parse/encode errors are silently caught and mapped
            to ``False``).

        Raises
        ------
        SolverTimeoutError
            If Z3 returns ``unknown`` (typically due to timeout).
        """
        try:
            sig = formula.signature
            if theory:
                sig = self._merge_theory_signature(sig, theory)

            ctx = Z3ContextBuilder(sig)
            symbols = ctx.build_symbols()
            encoder = FOLZ3Encoder(sort=ctx.sort)

            s = Solver()
            t = self._resolve_timeout(timeout)
            if t is not None:
                s.set('timeout', t)

            theory_trees = self._parse_theory_trees(theory or [])
            if self._uses_equality(formula.tree, *theory_trees):
                self._add_equality_axioms(s, ctx.sort)

            for z3_ax in self._encode_theory(theory or [], symbols, encoder):
                s.add(z3_ax)

            s.add(encoder.encode(formula.tree, symbols))

            return self._check(s, t)

        except SolverTimeoutError:
            raise
        except Exception:
            return False

    def are_equivalent(
        self,
        f1,
        f2,
        theory: list[str] | None = None,
        timeout: int | None = None,
    ) -> bool:
        """Return ``True`` if *f1* and *f2* are logically equivalent.

        Equivalence is checked by asserting ``f1 ≠ f2`` (i.e., ``f1 XOR f2``
        is satisfiable) and verifying that the result is *unsatisfiable*.

        Parameters
        ----------
        f1, f2 : FOL
            Formulas to compare.
        theory : list[str], optional
            Background axioms added before the check.
        timeout : int, optional
            Timeout in milliseconds.

        Returns
        -------
        bool
            ``True`` iff equivalent.  ``False`` on parse/encode errors.

        Raises
        ------
        SolverTimeoutError
            If Z3 returns ``unknown``.
        """
        try:
            sig = FOLSignature().merge(f1.signature, f2.signature)
            if theory:
                sig = self._merge_theory_signature(sig, theory)

            ctx = Z3ContextBuilder(sig)
            symbols = ctx.build_symbols()
            encoder = FOLZ3Encoder(sort=ctx.sort)

            s = Solver()
            t = self._resolve_timeout(timeout)
            if t is not None:
                s.set('timeout', t)

            theory_trees = self._parse_theory_trees(theory or [])
            if self._uses_equality(f1.tree, f2.tree, *theory_trees):
                self._add_equality_axioms(s, ctx.sort)

            for z3_ax in self._encode_theory(theory or [], symbols, encoder):
                s.add(z3_ax)

            z3_f1 = encoder.encode(f1.tree, symbols)
            z3_f2 = encoder.encode(f2.tree, symbols)
            # Equivalence ⟺ ¬∃model.(f1 ⊕ f2)  ⟺  (f1 ≠ f2) is unsat
            s.add(z3_f1 != z3_f2)

            # Unsat ⟹ equivalent
            return not self._check(s, t)

        except SolverTimeoutError:
            raise
        except Exception:
            return False

    def implies(
        self,
        f1,
        f2,
        theory: list[str] | None = None,
        timeout: int | None = None,
    ) -> bool:
        """Return ``True`` if *f1* logically entails *f2*.

        Entailment is checked by asserting ``f1 ∧ ¬f2`` and verifying
        unsatisfiability.

        Parameters
        ----------
        f1, f2 : FOL
            Premise and conclusion.
        theory : list[str], optional
            Background axioms.
        timeout : int, optional
            Timeout in milliseconds.

        Returns
        -------
        bool
            ``True`` iff *f1* ⊨ *f2*.  ``False`` on errors.

        Raises
        ------
        SolverTimeoutError
            If Z3 returns ``unknown``.
        """
        try:
            sig = FOLSignature().merge(f1.signature, f2.signature)
            if theory:
                sig = self._merge_theory_signature(sig, theory)

            ctx = Z3ContextBuilder(sig)
            symbols = ctx.build_symbols()
            encoder = FOLZ3Encoder(sort=ctx.sort)

            s = Solver()
            t = self._resolve_timeout(timeout)
            if t is not None:
                s.set('timeout', t)

            theory_trees = self._parse_theory_trees(theory or [])
            if self._uses_equality(f1.tree, f2.tree, *theory_trees):
                self._add_equality_axioms(s, ctx.sort)

            for z3_ax in self._encode_theory(theory or [], symbols, encoder):
                s.add(z3_ax)

            z3_f1 = encoder.encode(f1.tree, symbols)
            z3_f2 = encoder.encode(f2.tree, symbols)
            # Entailment ⟺  f1 ∧ ¬f2  is unsat
            s.add(z3_f1)
            s.add(Not(z3_f2))

            return not self._check(s, t)

        except SolverTimeoutError:
            raise
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers — solver execution
    # ------------------------------------------------------------------

    def _resolve_timeout(self, timeout: int | None) -> int | None:
        """Return the effective timeout: explicit arg > instance default."""
        if timeout is not None:
            return timeout
        return self._default_timeout

    def _check(self, solver: Solver, timeout: int | None) -> bool:
        """Run ``solver.check()`` and map the result to bool.

        Returns
        -------
        bool
            ``True`` if satisfiable.

        Raises
        ------
        SolverTimeoutError
            If Z3 returns ``unknown``.
        """
        result = solver.check()
        if result == sat:
            return True
        if result == unsat:
            return False
        raise SolverTimeoutError(
            f"Z3 returned '{result}' — likely a timeout "
            f"(limit: {timeout} ms)"
        )

    # ------------------------------------------------------------------
    # Internal helpers — signature and theory handling
    # ------------------------------------------------------------------

    def _merge_theory_signature(self, base_sig: dict, theory_strs: list[str]) -> dict:
        """Merge *base_sig* with the signatures of all *theory_strs*."""
        from .parser import FOLParser
        extractor = FOLSignature()
        parser = FOLParser()
        merged = base_sig
        for s in theory_strs:
            tree = parser.parse(s)
            sig = extractor.extract(tree)
            merged = extractor.merge(merged, sig)
        return merged

    def _parse_theory_trees(self, theory_strs: list[str]) -> list:
        """Parse each axiom string and return a list of AST roots."""
        from .parser import FOLParser
        parser = FOLParser()
        return [parser.parse(s) for s in theory_strs]

    def _encode_theory(self, theory_strs: list[str], symbols: dict, encoder: FOLZ3Encoder) -> list:
        """Parse and encode every axiom string into Z3 expressions."""
        from .parser import FOLParser
        parser = FOLParser()
        return [encoder.encode(parser.parse(s), symbols) for s in theory_strs]

    # ------------------------------------------------------------------
    # Internal helpers — equality axioms
    # ------------------------------------------------------------------

    def _uses_equality(self, *nodes) -> bool:
        """Return ``True`` if any node in *nodes* contains a ``'='`` relation."""
        from .ast import RelationNode, BooleanNode, QuantifierNode

        def _check(node) -> bool:
            if isinstance(node, RelationNode):
                return node.name == '='
            if isinstance(node, BooleanNode):
                return any(_check(c) for c in node.children)
            if isinstance(node, QuantifierNode):
                return _check(node.body)
            return False

        return any(_check(n) for n in nodes)

    def _add_equality_axioms(self, solver: Solver, sort) -> None:
        """Assert reflexivity, symmetry, and transitivity for *sort*.

        These axioms are formally redundant (Z3's ``==`` satisfies them
        natively) but make the equality theory fully explicit in the
        solver's assertion stack, which aids debugging.
        """
        from z3 import Const as Z3Const
        x = Z3Const('_eq_x', sort)
        y = Z3Const('_eq_y', sort)
        z = Z3Const('_eq_z', sort)
        solver.add(ForAll([x], x == x))
        solver.add(ForAll([x, y], Implies(x == y, y == x)))
        solver.add(ForAll([x, y, z], Implies(And(x == y, y == z), x == z)))
