"""
fol_tools.ast
=============

AST node types for First-Order Logic formulas.

Every parsed formula is represented as a tree of :class:`Node` objects.
The four concrete node kinds cover all syntactic forms recognised by this
library:

+-------------------+-------------------------------------------------+
| Node class        | Represents                                      |
+===================+=================================================+
| :class:`QuantifierNode` | ∀x φ  or  ∃x φ                        |
+-------------------+-------------------------------------------------+
| :class:`BooleanNode`    | ¬φ, φ∧ψ, φ∨ψ, φ→ψ, φ↔ψ, φ⊕ψ          |
+-------------------+-------------------------------------------------+
| :class:`RelationNode`   | R(t₁,…,tₙ)  or  t₁ = t₂               |
+-------------------+-------------------------------------------------+
| :class:`BoolConstNode`  | ⊤ (True)  /  ⊥ (False)                |
+-------------------+-------------------------------------------------+

Design notes
------------
* Nodes are **plain data containers** — they carry no logic.  All
  algorithms (parsing, encoding, translation, …) are implemented in
  separate modules that receive a root :class:`Node` and walk the tree.

* ``Node`` is intentionally kept as a minimal base class (no abstract
  methods) so that external code can subclass or duck-type it freely.

* Term arguments in :class:`RelationNode` are stored as plain strings
  (the original token) rather than a richer term type.  This keeps the
  AST small and easy to serialise; arities and sorts are tracked in the
  :mod:`fol_tools.signature` module.

Extending the AST
-----------------
To add a new node kind:

1. Create a subclass of :class:`Node` here.
2. Update :class:`~fol_tools.parser.FOLParser` to produce it.
3. Update every visitor in ``encoder``, ``translator``, ``signature``,
   ``perturbations``, and ``formula`` to handle the new type.
"""

from __future__ import annotations


class Node:
    """Abstract base class for all AST nodes.

    All concrete nodes inherit from this class.  No methods are defined
    here; the base class exists solely to give a common type that callers
    can use in ``isinstance`` checks and type annotations.
    """


class QuantifierNode(Node):
    """Represents a first-order quantifier application.

    Encodes both *universal* (∀) and *existential* (∃) quantification.

    Attributes
    ----------
    quantifier : {'forall', 'exists'}
        Which quantifier this node represents.
    variable : str
        The bound variable name (always a lowercase identifier).
    body : Node
        The sub-formula over which the quantifier ranges.

    Example
    -------
    The formula ``∀x Human(x)`` is represented as::

        QuantifierNode(
            quantifier='forall',
            variable='x',
            body=RelationNode(name='Human', arguments=['x'])
        )
    """

    def __init__(self, quantifier: str, variable: str, body: Node) -> None:
        if quantifier not in ('forall', 'exists'):
            raise ValueError(
                f"quantifier must be 'forall' or 'exists', got {quantifier!r}"
            )
        self.quantifier: str = quantifier
        self.variable: str = variable
        self.body: Node = body

    def __repr__(self) -> str:  # pragma: no cover
        q = '∀' if self.quantifier == 'forall' else '∃'
        return f"QuantifierNode({q}{self.variable}, {self.body!r})"


class BooleanNode(Node):
    """Represents a Boolean connective applied to one or two sub-formulas.

    Attributes
    ----------
    operator : {'not', 'and', 'or', 'implies', 'iff', 'xor'}
        The logical operator.
    children : list[Node]
        For ``'not'``: exactly one child.
        For all binary operators: exactly two children (left then right).

    Operator semantics
    ------------------
    =========  =======  ============================
    operator   symbol   semantics
    =========  =======  ============================
    not        ¬        logical negation
    and        ∧        conjunction
    or         ∨        disjunction
    implies    →        material implication (right-associative)
    iff        ↔        biconditional (⟺)
    xor        ⊕        exclusive disjunction
    =========  =======  ============================

    Example
    -------
    ``¬A(x)`` is::

        BooleanNode(operator='not', children=[RelationNode('A', ['x'])])
    """

    _VALID_OPERATORS = frozenset({'not', 'and', 'or', 'implies', 'iff', 'xor'})

    def __init__(self, operator: str, children: list[Node]) -> None:
        if operator not in self._VALID_OPERATORS:
            raise ValueError(
                f"operator must be one of {sorted(self._VALID_OPERATORS)}, "
                f"got {operator!r}"
            )
        self.operator: str = operator
        self.children: list[Node] = list(children)

    def __repr__(self) -> str:  # pragma: no cover
        return f"BooleanNode({self.operator!r}, {self.children!r})"


class RelationNode(Node):
    """Represents a relational atom: R(t₁, …, tₙ) or t₁ = t₂.

    Attributes
    ----------
    name : str
        Relation symbol name.  For equality this is ``'='``; all other
        relations start with an uppercase letter (by convention).
    arguments : list[str]
        Ordered list of term tokens.  Terms are variable names (lowercase),
        constant names (uppercase or digit-starting), or the special
        numeric tokens produced by the tokeniser.

    Arity
    -----
    ``len(arguments)`` gives the arity.  A zero-arity relation is a
    propositional atom (Boolean constant treated as a relation, e.g.
    ``Rain`` with no arguments).

    Built-in infix predicates
    -------------------------
    The parser produces ``RelationNode`` for the following infix atoms:

    * ``t₁ = t₂``  →  ``RelationNode('=',  [t₁, t₂])``
    * ``t₁ ≠ t₂``  →  ``BooleanNode('not', [RelationNode('=', [t₁, t₂])])``
    * ``t₁ < t₂``  →  ``RelationNode('<',  [t₁, t₂])``
    * ``t₁ > t₂``  →  ``RelationNode('>',  [t₁, t₂])``

    All three predicate names (``'='``, ``'<'``, ``'>'``) appear in
    the signature's ``Rel`` dict with arity 2.  The encoder maps ``'='``
    to Z3's native ``==`` (congruence axioms hold automatically); ``'<'``
    and ``'>'`` are encoded as uninterpreted binary functions by default.

    Example
    -------
    ``Loves(x, John)`` →  ``RelationNode(name='Loves', arguments=['x', 'John'])``

    ``x = y``           →  ``RelationNode(name='=', arguments=['x', 'y'])``
    """

    def __init__(self, name: str, arguments: list[str]) -> None:
        self.name: str = name
        self.arguments: list[str] = list(arguments)

    def __repr__(self) -> str:  # pragma: no cover
        if self.arguments:
            return f"RelationNode({self.name!r}, {self.arguments!r})"
        return f"RelationNode({self.name!r})"


class BoolConstNode(Node):
    """Represents a Boolean constant: ⊤ (True) or ⊥ (False).

    These correspond to tautology (⊤ / ``True``) and contradiction
    (⊥ / ``False``) in the formula language.

    Attributes
    ----------
    value : bool
        ``True`` for ⊤/True, ``False`` for ⊥/False.

    Example
    -------
    The formula ``⊤`` → ``BoolConstNode(value=True)``
    """

    def __init__(self, value: bool) -> None:
        self.value: bool = bool(value)

    def __repr__(self) -> str:  # pragma: no cover
        return f"BoolConstNode({'⊤' if self.value else '⊥'})"
