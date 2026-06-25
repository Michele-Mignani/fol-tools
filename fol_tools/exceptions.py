"""
fol_tools.exceptions
====================

All public exception classes for fol_tools.

Hierarchy
---------
::

    Exception
    └── SolverTimeoutError      Z3 exceeded the timeout budget

    ValueError
    ├── FOLSyntaxError          Malformed formula string or invalid AST construction
    ├── FOLEncoderError         Unknown AST node / operator during Z3 encoding or translation
    └── FOLSignatureError       Arity conflict when merging two signatures
"""

from __future__ import annotations


class SolverTimeoutError(Exception):
    """Raised when a Z3 solver call exceeds the specified timeout.

    Raised by :meth:`~fol_tools.solver.FOLSolver.implies`,
    :meth:`~fol_tools.solver.FOLSolver.are_equivalent`, and
    :meth:`~fol_tools.solver.FOLSolver.is_satisfiable` when Z3 returns
    ``unknown`` (typically because the per-call timeout was exceeded).
    """


class FOLSyntaxError(ValueError):
    """Raised when a formula string or AST construction is malformed.

    Covers:

    * Forbidden ASCII operators (``->``, ``&``) in a formula string.
    * Unbalanced or missing parentheses.
    * Unexpected / unrecognised tokens.
    * Empty formula string.
    * Lowercase standalone relation names (without a following ``(``).
    * Invalid quantifier / operator values passed to AST node constructors.
    * Any syntax error caught by :class:`~fol_tools.parser.FOLParser` and
      re-raised by the solver methods.

    This is also what :meth:`~fol_tools.solver.FOLSolver.implies` and
    :meth:`~fol_tools.solver.FOLSolver.are_equivalent` raise when they
    encounter a parse error in a formula or theory string, allowing callers
    to distinguish *syntax errors* from *logical non-entailment*
    (:meth:`~fol_tools.solver.FOLSolver.implies` returning ``False``) and
    *solver timeouts* (:exc:`SolverTimeoutError`).
    """


class FOLEncoderError(ValueError):
    """Raised when an AST node or boolean operator is unknown during encoding.

    Covers:

    * :class:`~fol_tools.encoder.FOLZ3Encoder` encountering an AST node type
      it cannot encode into a Z3 expression.
    * :class:`~fol_tools.encoder.FOLZ3Encoder` encountering an unknown boolean
      operator string.
    * :class:`~fol_tools.translator.FOLTranslator` encountering an unknown node
      type during string or natural-language translation.

    These errors indicate an internal invariant violation (e.g. a new AST node
    type was added without updating the encoder) rather than a user-facing
    formula syntax error.
    """


class FOLSignatureError(ValueError):
    """Raised when two signatures cannot be merged due to an arity conflict.

    Raised by :meth:`~fol_tools.signature.FOLSignature.merge` when the same
    relation name appears in both signatures with different arities.
    """


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: Deprecated alias for :exc:`FOLSyntaxError`.  Will be removed in a future
#: version.  Import :exc:`FOLSyntaxError` directly instead.
FormulaParseError = FOLSyntaxError