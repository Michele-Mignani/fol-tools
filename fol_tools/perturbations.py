"""
fol_tools.perturbations
=======================

Rule-based structural perturbations for FOL formulas.

A *perturbation* is a semantically distinct formula obtained by applying
a small, deterministic structural transformation to a given formula.
Perturbations are useful for:

* generating near-miss variants in dataset curation (e.g. to produce
  plausible incorrect alternatives);
* stress-testing reasoners with logically-close pairs;
* data augmentation.

Available transforms
--------------------
Each transform is a function ``Node -> Node | None``.  ``None`` means
the transform does not apply to the given formula.  Transforms never
mutate their input; they return a fresh ``Node`` tree built with
``copy.deepcopy``.

+-----------------------------------------+----------------------------------------------+
| Function                                | Effect                                       |
+=========================================+==============================================+
| :func:`flip_root_quantifier`            | ∀ ↔ ∃ at the outermost quantifier            |
+-----------------------------------------+----------------------------------------------+
| :func:`flip_root_connective`            | → ↔ ↔, ∧ ↔ ∨ at the first binary connective |
+-----------------------------------------+----------------------------------------------+
| :func:`negate_whole`                    | Wrap formula in ¬(…)                         |
+-----------------------------------------+----------------------------------------------+
| :func:`negate_consequent`               | A → B  becomes  A → ¬B                       |
+-----------------------------------------+----------------------------------------------+
| :func:`swap_nested_quantifiers`         | ∀x∃y φ  becomes  ∃y∀x φ                      |
+-----------------------------------------+----------------------------------------------+

High-level helper
-----------------
:func:`generate_rule_based` applies all transforms in order and returns
up to *n* distinct perturbations as formula strings.

Usage
-----
.. code-block:: python

    from fol_tools.perturbations import generate_rule_based

    variants = generate_rule_based("∀x (Human(x) → Mortal(x))", n=4)
    # ['∃x Human(x) → Mortal(x)',
    #  '∀x (Human(x) ↔ Mortal(x))',
    #  '∀x (Human(x) → ¬Mortal(x))',
    #  '¬∀x Human(x) → Mortal(x)']

Extending
---------
To add a new transform:

1. Write a function ``my_transform(node: Node) -> Node | None`` in this
   module.
2. Append it to ``RULE_TRANSFORMS``.

The :func:`generate_rule_based` function will pick it up automatically.
"""

from __future__ import annotations

import copy
from typing import Callable

from .ast import Node, QuantifierNode, BooleanNode, RelationNode


# ------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------

def _clone(node: Node) -> Node:
    """Return a deep copy of *node* (does not modify the original)."""
    return copy.deepcopy(node)


# ------------------------------------------------------------------
# Individual transforms
# ------------------------------------------------------------------

def flip_root_quantifier(node: Node) -> Node | None:
    """Flip the outermost quantifier: ∀ → ∃ or ∃ → ∀.

    Parameters
    ----------
    node : Node
        Root of the formula AST.

    Returns
    -------
    Node or None
        A new ``QuantifierNode`` with the flipped quantifier, or ``None``
        if the root is not a quantifier.

    Example
    -------
    ``∀x Human(x)``  →  ``∃x Human(x)``
    """
    if isinstance(node, QuantifierNode):
        new_q = 'exists' if node.quantifier == 'forall' else 'forall'
        return QuantifierNode(new_q, node.variable, _clone(node.body))
    return None


def flip_root_connective(node: Node) -> Node | None:
    """Flip the first binary connective found (descending through quantifiers).

    Mappings applied:

    * ``→`` ↔ ``↔``
    * ``∧`` ↔ ``∨``

    Unary ``not`` and quantifier layers are passed through transparently.
    Returns ``None`` if no applicable connective is found.

    Parameters
    ----------
    node : Node
        Root of the formula AST.

    Returns
    -------
    Node or None
        A new tree with the first applicable connective flipped, or ``None``.

    Example
    -------
    ``∀x (A(x) → B(x))``  →  ``∀x (A(x) ↔ B(x))``
    """
    _FLIPS: dict[str, str] = {
        'implies': 'iff',
        'iff': 'implies',
        'and': 'or',
        'or': 'and',
    }

    if isinstance(node, QuantifierNode):
        result = flip_root_connective(node.body)
        if result is None:
            return None
        return QuantifierNode(node.quantifier, node.variable, result)

    if isinstance(node, BooleanNode) and node.operator in _FLIPS:
        return BooleanNode(_FLIPS[node.operator], [_clone(c) for c in node.children])

    return None


def negate_whole(node: Node) -> Node:
    """Wrap the entire formula in ¬(…).

    Parameters
    ----------
    node : Node
        Root of the formula AST.

    Returns
    -------
    Node
        ``BooleanNode('not', [clone(node)])``.

    Note
    ----
    This transform always produces a result (never returns ``None``).

    Example
    -------
    ``∀x Human(x)``  →  ``¬∀x Human(x)``
    """
    return BooleanNode('not', [_clone(node)])


def negate_consequent(node: Node) -> Node | None:
    """In an implication A → B (under any outer quantifiers), produce A → ¬B.

    The transform descends through leading quantifiers to find the first
    implication node.  Returns ``None`` if no implication is found.

    Parameters
    ----------
    node : Node
        Root of the formula AST.

    Returns
    -------
    Node or None
        Tree with the consequent negated, or ``None`` if not applicable.

    Example
    -------
    ``∀x (Human(x) → Mortal(x))``  →  ``∀x (Human(x) → ¬Mortal(x))``
    """
    if isinstance(node, QuantifierNode):
        result = negate_consequent(node.body)
        if result is None:
            return None
        return QuantifierNode(node.quantifier, node.variable, result)

    if isinstance(node, BooleanNode) and node.operator == 'implies':
        ante = _clone(node.children[0])
        cons = BooleanNode('not', [_clone(node.children[1])])
        return BooleanNode('implies', [ante, cons])

    return None


def swap_nested_quantifiers(node: Node) -> Node | None:
    """Swap the first two nested quantifiers.

    ``∀x ∃y φ``  →  ``∃y ∀x φ``

    Works only if the outermost node is a quantifier whose immediate body
    is also a quantifier.

    Parameters
    ----------
    node : Node
        Root of the formula AST.

    Returns
    -------
    Node or None
        Tree with the two outermost quantifiers swapped, or ``None`` if
        the structure does not match (fewer than two nested quantifiers).

    Example
    -------
    ``∀x ∃y Loves(x, y)``  →  ``∃y ∀x Loves(x, y)``
    """
    if (
        isinstance(node, QuantifierNode)
        and isinstance(node.body, QuantifierNode)
    ):
        outer, inner = node, node.body
        new_inner = QuantifierNode(outer.quantifier, outer.variable, _clone(inner.body))
        return QuantifierNode(inner.quantifier, inner.variable, new_inner)
    return None


# ------------------------------------------------------------------
# Registry of all rule-based transforms (ordered)
# ------------------------------------------------------------------

RULE_TRANSFORMS: list[Callable[[Node], Node | None]] = [
    flip_root_quantifier,
    flip_root_connective,
    negate_consequent,
    swap_nested_quantifiers,
    negate_whole,
]
"""Ordered list of all rule-based transforms applied by :func:`generate_rule_based`.

To add a new transform without touching :func:`generate_rule_based`, append to this list.
"""


# ------------------------------------------------------------------
# High-level generator
# ------------------------------------------------------------------

def generate_rule_based(formula_str: str, n: int) -> list[str]:
    """Apply all transforms in :data:`RULE_TRANSFORMS` and return up to *n* results.

    Transforms are applied in the order they appear in
    :data:`RULE_TRANSFORMS`.  Duplicates (including the original formula)
    are discarded.  Transforms that fail or do not apply are silently
    skipped.

    Parameters
    ----------
    formula_str : str
        The input formula in Unicode FOL syntax.
    n : int
        Maximum number of perturbations to return.

    Returns
    -------
    list[str]
        Distinct perturbation strings, length ≤ *n*.  May be empty if
        the input fails to parse or no transform produces a new result.

    Examples
    --------
    .. code-block:: python

        >>> generate_rule_based("∀x (Human(x) → Mortal(x))", n=3)
        ['∃x Human(x) → Mortal(x)',
         '∀x (Human(x) ↔ Mortal(x))',
         '∀x (Human(x) → ¬Mortal(x))']
    """
    from .parser import FOLParser
    from .translator import FOLTranslator

    try:
        tree = FOLParser().parse(formula_str)
    except Exception:
        return []

    translator = FOLTranslator()
    results: list[str] = []
    seen: set[str] = {formula_str}

    for transform in RULE_TRANSFORMS:
        if len(results) >= n:
            break
        try:
            modified = transform(tree)
            if modified is None:
                continue
            s = translator.to_string(modified)
            if s not in seen:
                results.append(s)
                seen.add(s)
        except Exception:
            continue

    return results
