"""
fol_tools.signature
===================

Signature extraction and merging for FOL formulas.

A *logical signature* is the catalogue of non-logical symbols that appear
in a formula (or a set of formulas).  It is represented as a plain
dictionary with three keys:

.. code-block:: python

    {
        'Rel':   {'Human': 1, 'Loves': 2},   # name → arity
        'Var':   {'x', 'y'},                  # quantifier-bound variables
        'Const': {'John', '3'},               # constants (uppercase or numeric)
    }

The signature is needed by :mod:`fol_tools.encoder` to instantiate Z3
symbols before encoding a formula.

Symbol classification rules
----------------------------
* **Relation** (``Rel``): any ``RelationNode.name``, including the
  built-in equality ``'='`` and the comparison predicates ``'<'`` and
  ``'>'``.  Arity = ``len(RelationNode.arguments)``.
* **Variable** (``Var``): a term that matches the pattern ``[a-z][0-9]*``
  (a single lowercase letter, optionally followed by digits — e.g. ``x``,
  ``y``, ``x1``, ``z2``) **and** is bound by an enclosing quantifier.
* **Constant** (``Const``): everything else, specifically:
  - Uppercase-starting names (``John``, ``LowSurvivalRate``).
  - Digit-starting tokens (``3``, ``42``).
  - camelCase names with interior uppercase (``diamondMine``, ``wWE``).
  - Multi-character all-lowercase names (``tom``, ``joey``, ``istanbul``).
  - Single-letter lowercase names that are *free* (not bound by a quantifier).

Usage
-----
.. code-block:: python

    from fol_tools.parser import FOLParser
    from fol_tools.signature import FOLSignature

    tree = FOLParser().parse("∀x Loves(x, John)")
    sig = FOLSignature().extract(tree)
    # {'Rel': {'Loves': 2}, 'Var': {'x'}, 'Const': {'John'}}

Merging two signatures
----------------------
.. code-block:: python

    sig2 = FOLSignature().extract(FOLParser().parse("∀y Mortal(y)"))
    merged = FOLSignature().merge(sig, sig2)

``merge`` raises ``ValueError`` if the same relation name appears with
different arities in the two signatures.
"""

from __future__ import annotations

import re

from .ast import Node, QuantifierNode, BooleanNode, RelationNode, BoolConstNode
from .exceptions import FOLSignatureError

# A term token is treated as a *variable candidate* iff it matches this pattern:
# one lowercase ASCII letter optionally followed by one or more digits (x, y, x1, z2).
# Everything else (multi-char lowercase, camelCase, uppercase-starting, digit-starting)
# is classified as an individual constant.
_VAR_PATTERN = re.compile(r'^[a-z][0-9]*$')


class FOLSignature:
    """Extract and merge logical signatures from FOL ASTs.

    All methods are stateless; a single instance can be reused freely.

    Methods
    -------
    extract(tree)
        Walk an AST and return a fresh signature dict.
    merge(sig1, sig2)
        Combine two signature dicts, raising on arity conflicts.
    empty()
        Return a blank signature dict ``{'Rel': {}, 'Var': set(), 'Const': set()}``.
    """

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    @staticmethod
    def empty() -> dict:
        """Return an empty signature dictionary.

        Returns
        -------
        dict
            ``{'Rel': {}, 'Var': set(), 'Const': set()}``
        """
        return {'Rel': {}, 'Var': set(), 'Const': set()}

    def extract(self, tree: Node) -> dict:
        """Walk *tree* and extract all non-logical symbols.

        Parameters
        ----------
        tree : Node
            Root of a parsed FOL AST.

        Returns
        -------
        dict
            A new signature dict ``{'Rel': {…}, 'Var': {…}, 'Const': {…}}``.
            The ``Rel`` sub-dict maps each relation name to its arity (int).
            ``Var`` and ``Const`` are plain sets of strings.
        """
        sig = self.empty()
        self._walk(tree, sig, bound=frozenset())
        return sig

    def merge(self, sig1: dict, sig2: dict) -> dict:
        """Return a new signature that is the union of *sig1* and *sig2*.

        Parameters
        ----------
        sig1, sig2 : dict
            Signature dicts as produced by :meth:`extract`.

        Returns
        -------
        dict
            Merged signature.

        Raises
        ------
        ValueError
            If the same relation name appears with different arities in
            *sig1* and *sig2*.
        """
        merged = {
            'Rel': dict(sig1['Rel']),
            'Var': set(sig1['Var']) | set(sig2['Var']),
            'Const': set(sig1['Const']) | set(sig2['Const']),
        }
        for name, arity in sig2['Rel'].items():
            if name in merged['Rel'] and merged['Rel'][name] != arity:
                raise ValueError(
                    f"Arity conflict for relation '{name}': "
                    f"arity {merged['Rel'][name]} in sig1 vs "
                    f"arity {arity} in sig2"
                )
            merged['Rel'][name] = arity
        return merged

    # ------------------------------------------------------------------
    # Internal tree-walking
    # ------------------------------------------------------------------

    def _walk(self, node: Node, sig: dict, bound: frozenset) -> None:
        """Recursively populate *sig* by walking *node*.

        Parameters
        ----------
        node : Node
            Current AST node.
        sig : dict
            Signature dict being populated (mutated in place).
        bound : frozenset[str]
            Set of variable names currently in quantifier scope.
        """
        if isinstance(node, BoolConstNode):
            return  # ⊤/⊥ contribute no symbols

        if isinstance(node, QuantifierNode):
            # The quantifier's variable joins the bound set for the body.
            self._walk(node.body, sig, bound | {node.variable})

        elif isinstance(node, BooleanNode):
            for child in node.children:
                self._walk(child, sig, bound)

        elif isinstance(node, RelationNode):
            sig['Rel'][node.name] = len(node.arguments)

            for arg in node.arguments:
                if not _VAR_PATTERN.match(arg):
                    # Not a variable candidate (uppercase-starting, digit-starting,
                    # camelCase, or multi-char lowercase) → individual constant
                    sig['Const'].add(arg)
                elif arg in bound:
                    # Variable candidate ([a-z][0-9]*) in quantifier scope → variable
                    sig['Var'].add(arg)
                else:
                    # Variable candidate but free → treat as constant
                    sig['Const'].add(arg)
