"""
fol_tools
=========

A Python library for parsing, encoding, reasoning about, and translating
First-Order Logic (FOL) formulas written in Unicode syntax.

Quick start
-----------
.. code-block:: python

    from fol_tools import FOL, FOLSolver

    f1 = FOL("∀x (Human(x) → Mortal(x))")
    f2 = FOL("∀x (¬Mortal(x) → ¬Human(x))")

    solver = FOLSolver()
    solver.are_equivalent(f1, f2)   # True  — contrapositive

Package layout
--------------
.. list-table::
   :header-rows: 1

   * - Module
     - Provides
   * - :mod:`fol_tools.ast`
     - AST node classes (:class:`~fol_tools.ast.QuantifierNode`, etc.)
   * - :mod:`fol_tools.parser`
     - :class:`~fol_tools.parser.FOLParser` — string → AST
   * - :mod:`fol_tools.signature`
     - :class:`~fol_tools.signature.FOLSignature` — symbol extraction
   * - :mod:`fol_tools.encoder`
     - :class:`~fol_tools.encoder.Z3ContextBuilder`, :class:`~fol_tools.encoder.FOLZ3Encoder`
   * - :mod:`fol_tools.solver`
     - :class:`~fol_tools.solver.FOLSolver`, :exc:`~fol_tools.solver.SolverTimeoutError`
   * - :mod:`fol_tools.translator`
     - :class:`~fol_tools.translator.FOLTranslator` — AST → string / NL
   * - :mod:`fol_tools.perturbations`
     - Rule-based formula perturbations
   * - :mod:`fol_tools.formula`
     - :class:`~fol_tools.formula.FOL` — high-level façade

Public API
----------
The names exported from this top-level ``__init__`` are the recommended
public interface.  Lower-level classes (e.g. individual AST node types)
can be imported directly from their modules when needed.
"""

from .formula import FOL
from .parser import FOLParser
from .signature import FOLSignature
from .encoder import Z3ContextBuilder, FOLZ3Encoder
from .solver import FOLSolver, SolverTimeoutError
from .translator import FOLTranslator, DEFAULT_NL_RULES
from .perturbations import generate_rule_based, RULE_TRANSFORMS
from .ast import (
    Node,
    QuantifierNode,
    BooleanNode,
    RelationNode,
    BoolConstNode,
)

__all__ = [
    # High-level façade
    'FOL',
    # Core components
    'FOLParser',
    'FOLSignature',
    'Z3ContextBuilder',
    'FOLZ3Encoder',
    'FOLSolver',
    'SolverTimeoutError',
    'FOLTranslator',
    'DEFAULT_NL_RULES',
    # Perturbations
    'generate_rule_based',
    'RULE_TRANSFORMS',
    # AST node types
    'Node',
    'QuantifierNode',
    'BooleanNode',
    'RelationNode',
    'BoolConstNode',
]

__version__ = '0.1.0'
