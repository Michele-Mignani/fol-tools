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
* **Relation** (``Rel``): any ``RelationNode.name`` that is not the
  built-in equality ``'='``.  Arity = ``len(RelationNode.arguments)``.
* **Variable** (``Var``): a lowercase argument that appears *inside* the
  scope of a quantifier that binds it.
* **Constant** (``Const``): an uppercase argument, a digit-starting token,
  or a *free* lowercase name (an argument not bound by any enclosing
  quantifier).

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

from .ast import Node, QuantifierNode, BooleanNode, RelationNode, BoolConstNode


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
            # '=' is built-in; do not catalogue it as a user-defined relation.
            if node.name != '=':
                sig['Rel'][node.name] = len(node.arguments)

            for arg in node.arguments:
                if arg[0].isupper():
                    # Uppercase → individual constant
                    sig['Const'].add(arg)
                elif arg[0].isdigit():
                    # Digit-starting numeric token → constant
                    sig['Const'].add(arg)
                elif arg in bound:
                    # Lowercase and currently in scope → variable
                    sig['Var'].add(arg)
                else:
                    # Lowercase but free → treat as a constant term
                    sig['Const'].add(arg)
