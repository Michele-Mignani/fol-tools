"""
fol_tools.encoder
=================

Z3 context builder and AST-to-Z3 encoder.

This module bridges the FOL AST representation and the Z3 Python API.
It is used internally by :mod:`fol_tools.solver` and by
:meth:`~fol_tools.formula.FOL.to_smtlib`.

Architecture
------------
Two classes work together:

:class:`Z3ContextBuilder`
    Builds a dictionary of Z3 symbols (``Function`` objects and ``Const``
    objects) from a logical signature dict.  Think of it as the *context*
    — the mapping from string names to their Z3 counterparts.

:class:`FOLZ3Encoder`
    Walks a FOL AST and recursively constructs the corresponding Z3
    expression, using the symbol dictionary produced by
    :class:`Z3ContextBuilder`.

Domain
------
The encoder uses a *single uninterpreted sort* ``U`` to represent all
first-order individuals.  This is the standard approach for classical
first-order logic over an unspecified domain.  It means:

* All individual constants and variables have sort ``U``.
* An *n*-ary relation ``R`` is encoded as a ``Function U × … × U → Bool``.
* A zero-arity relation is encoded as a ``Const`` of sort ``Bool``
  (a propositional atom).

Equality and comparison predicates
------------------------------------
``'='`` appears in the signature as a regular arity-2 relation but is
mapped to Z3's native ``==`` operator by the encoder (this gives
reflexivity, symmetry, transitivity, and congruence for free).

``'<'`` and ``'>'`` are uninterpreted binary predicate symbols by default
— the encoder creates a ``Function U × U → Bool`` for them just like any
other relation.  To get arithmetic or total-order semantics, assert the
relevant axioms via the *theory* parameter of :class:`~fol_tools.solver.FOLSolver`.

Usage
-----
.. code-block:: python

    from fol_tools.parser import FOLParser
    from fol_tools.signature import FOLSignature
    from fol_tools.encoder import Z3ContextBuilder, FOLZ3Encoder

    tree = FOLParser().parse("∀x (Human(x) → Mortal(x))")
    sig  = FOLSignature().extract(tree)

    ctx     = Z3ContextBuilder(sig)
    symbols = ctx.build_symbols()
    encoder = FOLZ3Encoder()

    z3_expr = encoder.encode(tree, symbols)
    print(z3_expr.sexpr())    # SMT-LIB string

Extending
---------
To support multiple sorts, replace the single ``DeclareSort("U")`` call
with a sort-inference step, then propagate sort information through the
symbol dictionary.  The encoder itself requires no changes as long as
``symbols[name]`` resolves to a correctly-typed Z3 object.
"""

from __future__ import annotations

from z3 import (
    BoolSort,
    BoolVal,
    Const,
    DeclareSort,
    Exists,
    ForAll,
    Function,
    And,
    Or,
    Not,
    Implies,
    Xor,
)

from .ast import Node, QuantifierNode, BooleanNode, RelationNode, BoolConstNode
from .exceptions import FOLEncoderError


class Z3ContextBuilder:
    """Build a dictionary of Z3 symbols from a logical signature.

    A single uninterpreted sort ``U`` is used for all individuals.

    Parameters
    ----------
    signature : dict
        Signature dict ``{'Rel': {name: arity}, 'Var': {…}, 'Const': {…}}``,
        as produced by :class:`~fol_tools.signature.FOLSignature`.

    Attributes
    ----------
    sort : z3.SortRef
        The uninterpreted sort ``U`` shared by all symbols in this context.
        Exposed so that :mod:`fol_tools.solver` can add equality axioms.
    """

    def __init__(self, signature: dict) -> None:
        self.signature = signature
        self.sort = DeclareSort('U')

    def build_symbols(self) -> dict:
        """Instantiate Z3 symbol objects for every name in the signature.

        Returns
        -------
        dict
            Maps each relation name and constant name to its Z3 object:

            * *n*-ary relation (n ≥ 1): ``z3.Function(name, U, …, U, BoolSort())``
            * 0-ary relation (propositional atom): ``z3.Const(name, BoolSort())``
            * individual constant: ``z3.Const(name, U)``

        Notes
        -----
        Variable names are **not** pre-allocated here; the encoder creates
        fresh ``Const`` objects for quantifier-bound variables on the fly
        when it encounters a :class:`~fol_tools.ast.QuantifierNode`.
        """
        symbols: dict = {}
        U = self.sort

        for name, arity in self.signature['Rel'].items():
            if arity == 0:
                symbols[name] = Const(name, BoolSort())
            else:
                symbols[name] = Function(name, *([U] * arity + [BoolSort()]))

        for name in self.signature['Const']:
            symbols[name] = Const(name, U)

        return symbols


class FOLZ3Encoder:
    """Recursively encode a FOL AST into a Z3 expression.

    The encoder must be used together with a symbol dictionary produced
    by :meth:`Z3ContextBuilder.build_symbols`.

    Parameters
    ----------
    sort : z3.SortRef, optional
        The uninterpreted sort for individuals.  If omitted, a fresh
        ``DeclareSort('U')`` is created.  Pass the ``ctx.sort`` attribute
        of the :class:`Z3ContextBuilder` you used so that the sort objects
        are the *same* Python objects (Z3 identifies sorts by object
        identity in the same context).

    Usage
    -----
    .. code-block:: python

        ctx     = Z3ContextBuilder(sig)
        symbols = ctx.build_symbols()
        encoder = FOLZ3Encoder(sort=ctx.sort)
        z3_expr = encoder.encode(tree, symbols)
    """

    def __init__(self, sort=None) -> None:
        self.sort = sort if sort is not None else DeclareSort('U')

    def encode(self, node: Node, symbols: dict):
        """Translate *node* into a Z3 expression.

        Parameters
        ----------
        node : Node
            Root (or sub-root) of a FOL AST.
        symbols : dict
            Name → Z3 object mapping produced by
            :meth:`Z3ContextBuilder.build_symbols`.  The encoder
            **extends** this dictionary locally when it encounters
            quantifier-bound variables (by creating fresh ``Const``
            objects), but it never mutates the original dict.

        Returns
        -------
        z3.ExprRef
            A Z3 Boolean expression corresponding to *node*.

        Raises
        ------
        ValueError
            If *node* is of an unknown type, or if a symbol name is not
            found in *symbols* (usually indicates a missing constant or
            relation in the signature).
        """
        if isinstance(node, BoolConstNode):
            return BoolVal(node.value)
        if isinstance(node, QuantifierNode):
            return self._encode_quantifier(node, symbols)
        if isinstance(node, BooleanNode):
            return self._encode_boolean(node, symbols)
        if isinstance(node, RelationNode):
            return self._encode_relation(node, symbols)
        raise FOLEncoderError(f"Unknown AST node type: {type(node).__name__}")

    # ------------------------------------------------------------------
    # Node-specific encoding helpers
    # ------------------------------------------------------------------

    def _encode_quantifier(self, node: QuantifierNode, symbols: dict):
        """Encode ∀x φ or ∃x φ.

        Creates a fresh ``Const`` for the bound variable, injects it into
        a *copy* of *symbols*, and recursively encodes the body.
        """
        var = Const(node.variable, self.sort)
        local_symbols = {**symbols, node.variable: var}
        body = self.encode(node.body, local_symbols)
        if node.quantifier == 'forall':
            return ForAll([var], body)
        return Exists([var], body)

    def _encode_boolean(self, node: BooleanNode, symbols: dict):
        """Encode a Boolean connective node.

        Operator → Z3 function mapping:

        =========  ===========================================
        operator   Z3 encoding
        =========  ===========================================
        not        ``z3.Not(child)``
        and        ``z3.And(*children)``
        or         ``z3.Or(*children)``
        implies    ``z3.Implies(left, right)``
        iff        ``left == right``  (Z3 Boolean equality)
        xor        ``z3.Xor(left, right)``
        =========  ===========================================
        """
        children = [self.encode(c, symbols) for c in node.children]
        op = node.operator

        if op == 'not':
            return Not(children[0])
        if op == 'and':
            return And(*children)
        if op == 'or':
            return Or(*children)
        if op == 'implies':
            return Implies(children[0], children[1])
        if op == 'iff':
            # Z3 Boolean equality is biconditional for Bool expressions
            return children[0] == children[1]
        if op == 'xor':
            return Xor(children[0], children[1])
        raise FOLEncoderError(f"Unknown boolean operator: {op!r}")

    def _encode_relation(self, node: RelationNode, symbols: dict):
        """Encode a relational atom.

        Special cases:

        * ``'='`` is mapped to Z3's native ``==`` over sort ``U``, giving
          reflexivity, symmetry, transitivity, and congruence for free.
        * ``'<'`` and ``'>'`` are looked up in *symbols* as regular
          uninterpreted binary functions (created by :class:`Z3ContextBuilder`
          because they appear in the signature).
        * A zero-arity relation (propositional atom) returns the
          ``Bool``-sorted ``Const`` directly without calling it.
        * All other *n*-ary relations are applied to their encoded arguments.

        Raises
        ------
        KeyError
            If a symbol name does not appear in *symbols*, which means the
            signature passed to :class:`Z3ContextBuilder` was incomplete.
        """
        if node.name == '=':
            lhs = symbols[node.arguments[0]]
            rhs = symbols[node.arguments[1]]
            return lhs == rhs

        func = symbols[node.name]

        if not node.arguments:
            return func     # 0-arity propositional atom

        args = [symbols[arg] for arg in node.arguments]
        return func(*args)
