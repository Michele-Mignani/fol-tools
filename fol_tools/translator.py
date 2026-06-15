"""
fol_tools.translator
====================

AST-to-string serialisers for FOL formulas.

Three serialisation modes are provided:

:meth:`FOLTranslator.to_string`
    Serialise back to the canonical Unicode FOL syntax, with minimal
    parentheses (only those needed to override default precedence).

:meth:`FOLTranslator.to_tree_string`
    Serialise to a fully-parenthesised prefix tree notation, useful for
    debugging and for checking structural equality independent of
    operator precedence.

:meth:`FOLTranslator.to_nl`
    Rule-based natural-language generation.  Relations and constants are
    mapped to natural-language strings via a *symbol meanings* dictionary.
    All NL rules are overridable.

Usage
-----
.. code-block:: python

    from fol_tools.parser import FOLParser
    from fol_tools.translator import FOLTranslator

    tree = FOLParser().parse("∀x (Human(x) → Mortal(x))")
    tr = FOLTranslator()

    tr.to_string(tree)
    # '∀x Human(x) → Mortal(x)'

    tr.to_tree_string(tree)
    # 'Forall(x, Implies(Human(x), Mortal(x)))'

    tr.to_nl(tree, symbol_meanings={'Human': '{0} is human', 'Mortal': '{0} is mortal'})
    # 'For all x, if x is human, then x is mortal.'

Extending NL generation
-----------------------
Pass a custom ``rules`` dict to :meth:`to_nl` to override any of the
default templates listed in ``DEFAULT_NL_RULES``.  The dict is merged
with the defaults at call time, so you only need to specify what differs.

Symbol meanings
~~~~~~~~~~~~~~~
``symbol_meanings`` maps a relation or constant name to a template
string.  Argument slots are filled with ``str.format(*args)``.

    ``{'Loves': '{0} loves {1}', 'John': 'John'}``

When a relation has a registered *negative* meaning (via
``negative_meanings``), negated atoms ``¬R(…)`` are rendered using it
instead of the generic "it is not the case that …".
"""

from __future__ import annotations

from .ast import Node, QuantifierNode, BooleanNode, RelationNode, BoolConstNode

# ------------------------------------------------------------------
# Default NL rule templates
# ------------------------------------------------------------------

DEFAULT_NL_RULES: dict[str, str] = {
    'forall':  'for all {var}, {body}',
    'exists':  'there exists {var} such that {body}',
    'implies': 'if {0}, then {1}',
    'iff':     '{0} if and only if {1}',
    'xor':     'either {0} or {1} but not both',
    'and':     '{0} and {1}',
    'or':      '{0} or {1}',
    'not':     'it is not the case that {0}',
}

# ------------------------------------------------------------------
# Operator metadata for to_string
# ------------------------------------------------------------------

# Operator symbol → precedence (lower number = lower precedence / binds later)
_PRECEDENCE: dict[str, int] = {
    '↔': 1,   # iff
    '⊕': 2,   # xor
    '→': 3,   # implies
    '∨': 4,   # or
    '∧': 5,   # and
    '¬': 6,   # not
}

# Internal operator name → Unicode symbol
_OP_SYMBOL: dict[str, str] = {
    'iff':     '↔',
    'xor':     '⊕',
    'implies': '→',
    'or':      '∨',
    'and':     '∧',
    'not':     '¬',
}

# Operator names for to_tree_string
_TREE_OP: dict[str, str] = {
    'forall':  'Forall',
    'exists':  'Exists',
    'and':     'And',
    'or':      'Or',
    'implies': 'Implies',
    'iff':     'Iff',
    'xor':     'Xor',
    'not':     'Not',
}

# Binary operator names (not 'not')
_BINARY_OPS: frozenset[str] = frozenset({'and', 'or', 'implies', 'iff', 'xor'})

# NL precedence mirrors symbolic precedence exactly
_NL_PREC: dict[str, int] = {
    'iff':     1,
    'xor':     2,
    'implies': 3,
    'or':      4,
    'and':     5,
    'not':     6,
}


class FOLTranslator:
    """Serialise FOL AST nodes to strings or natural language.

    All methods are stateless; a single instance can be reused freely.
    """

    # ------------------------------------------------------------------
    # to_string — canonical Unicode FOL syntax
    # ------------------------------------------------------------------

    def to_string(self, node: Node, parent_prec: int = 0) -> str:
        """Serialise *node* to the canonical Unicode FOL string.

        Parentheses are inserted only when needed to preserve the intended
        operator precedence (i.e., when a child has lower precedence than
        its parent).

        Parameters
        ----------
        node : Node
            Root of the AST to serialise.
        parent_prec : int, optional
            Precedence of the enclosing operator (used internally for the
            recursive calls; callers should leave this at the default 0).

        Returns
        -------
        str
            Unicode formula string.

        Raises
        ------
        ValueError
            If *node* is of an unknown type.

        Examples
        --------
        >>> FOLTranslator().to_string(FOLParser().parse("∀x (Human(x) → Mortal(x))"))
        '∀x Human(x) → Mortal(x)'
        """
        if isinstance(node, BoolConstNode):
            return '⊤' if node.value else '⊥'

        if isinstance(node, QuantifierNode):
            q = '∀' if node.quantifier == 'forall' else '∃'
            body = self.to_string(node.body, parent_prec=0)
            return f"{q}{node.variable} {body}"

        if isinstance(node, BooleanNode):
            if node.operator == 'not':
                inner = self.to_string(node.children[0], parent_prec=_PRECEDENCE['¬'])
                return f"¬{inner}"
            sym = _OP_SYMBOL[node.operator]
            prec = _PRECEDENCE[sym]
            parts = [self.to_string(c, parent_prec=prec) for c in node.children]
            expr = f" {sym} ".join(parts)
            if prec < parent_prec:
                return f"({expr})"
            return expr

        if isinstance(node, RelationNode):
            if node.name in ('=', '<', '>') and len(node.arguments) == 2:
                return f"{node.arguments[0]} {node.name} {node.arguments[1]}"
            if node.arguments:
                return f"{node.name}({', '.join(node.arguments)})"
            return node.name

        raise ValueError(f"Unknown node type: {type(node).__name__}")

    # ------------------------------------------------------------------
    # to_tree_string — prefix tree notation
    # ------------------------------------------------------------------

    def to_tree_string(self, node: Node) -> str:
        """Serialise *node* to a fully-parenthesised prefix tree string.

        This format is unambiguous and independent of operator precedence.
        It is useful for debugging and for structural comparison.

        Parameters
        ----------
        node : Node
            Root of the AST.

        Returns
        -------
        str
            Prefix tree string, e.g. ``'Forall(x, Implies(Human(x), Mortal(x)))'``.

        Raises
        ------
        ValueError
            If *node* is of an unknown type.
        """
        if isinstance(node, BoolConstNode):
            return 'True' if node.value else 'False'

        if isinstance(node, QuantifierNode):
            name = _TREE_OP[node.quantifier]
            body = self.to_tree_string(node.body)
            return f"{name}({node.variable}, {body})"

        if isinstance(node, BooleanNode):
            name = _TREE_OP[node.operator]
            parts = ', '.join(self.to_tree_string(c) for c in node.children)
            return f"{name}({parts})"

        if isinstance(node, RelationNode):
            if node.arguments:
                return f"{node.name}({', '.join(node.arguments)})"
            return node.name

        raise ValueError(f"Unknown node type: {type(node).__name__}")

    # ------------------------------------------------------------------
    # to_nl — rule-based natural-language generation
    # ------------------------------------------------------------------

    def to_nl(
        self,
        node: Node,
        rules: dict[str, str] | None = None,
        symbol_meanings: dict[str, str] | None = None,
        negative_meanings: dict[str, str] | None = None,
        parenthesis: bool = False,
        split_conjunctions: bool = False,
    ) -> str:
        """Generate a natural-language sentence from *node*.

        Parameters
        ----------
        node : Node
            Root of the AST.
        rules : dict, optional
            Overrides for the NL templates in :data:`DEFAULT_NL_RULES`.
            Keys are operator names (``'forall'``, ``'implies'``, etc.).
            Only the entries you want to change need to be provided;
            the rest fall back to the defaults.
        symbol_meanings : dict, optional
            Maps relation/constant names to natural-language templates.
            Relation templates use positional ``{0}``, ``{1}``, … slots.
            Example: ``{'Human': '{0} is human', 'John': 'John'}``.
        negative_meanings : dict, optional
            Maps relation names to NL templates for the *negation* of
            that relation.  When a ``BooleanNode('not', [RelationNode(R, …)])``
            is encountered and ``R in negative_meanings``, the negative
            template is used instead of the generic "it is not the case that …".
        parenthesis : bool, optional
            If ``True``, wrap every binary sub-expression in parentheses
            regardless of precedence.  Default ``False``.
        split_conjunctions : bool, optional
            If ``True``, top-level conjuncts are separated with ``'. '``
            and each conjunct starts with a capital letter, producing
            multiple English sentences.  Default ``False``.

        Returns
        -------
        str
            An English sentence (capitalised, ending with a period).
            Returns an empty string if *node* produces no text.

        Raises
        ------
        ValueError
            If *node* is of an unknown type.
        """
        effective_rules = {**DEFAULT_NL_RULES, **(rules or {})}
        effective_meanings = dict(symbol_meanings or {})
        effective_negatives = dict(negative_meanings or {})

        raw = self._nl_node(
            node,
            effective_rules,
            effective_meanings,
            effective_negatives,
            parenthesis,
            split_conj=split_conjunctions,
            parent_op=None,
        )

        if not raw:
            return ''
        final = raw[0].upper() + raw[1:]
        if not final.endswith('.'):
            final += '.'
        return final

    # ------------------------------------------------------------------
    # Internal NL generation
    # ------------------------------------------------------------------

    def _nl_node(
        self,
        node: Node,
        rules: dict,
        meanings: dict,
        negative_meanings: dict,
        parenthesis: bool,
        split_conj: bool,
        parent_op: str | None,
    ) -> str:
        """Recursively generate NL for *node*.

        Parameters
        ----------
        node : Node
            Current node.
        rules : dict
            Effective (merged with defaults) NL rule templates.
        meanings : dict
            Symbol → NL template mapping.
        negative_meanings : dict
            Relation name → NL template for negated atoms.
        parenthesis : bool
            Force parentheses around every binary sub-expression.
        split_conj : bool
            Split top-level conjunctions into separate sentences.
        parent_op : str or None
            Operator name of the enclosing node (for precedence-driven
            parenthesisation).
        """
        if isinstance(node, BoolConstNode):
            return 'true' if node.value else 'false'

        # --- Quantifier ---
        if isinstance(node, QuantifierNode):
            variables = [node.variable]
            body_node = node.body
            # Collapse consecutive identical quantifiers: ∀x∀y → "for all x and y"
            while (
                isinstance(body_node, QuantifierNode)
                and body_node.quantifier == node.quantifier
            ):
                variables.append(body_node.variable)
                body_node = body_node.body

            body_nl = self._nl_node(
                body_node, rules, meanings, negative_meanings,
                parenthesis, split_conj=False, parent_op=None,
            )

            if len(variables) == 1:
                var_str = variables[0]
            elif len(variables) == 2:
                var_str = f"{variables[0]} and {variables[1]}"
            else:
                var_str = ', '.join(variables[:-1]) + f', and {variables[-1]}'

            tmpl = rules.get(node.quantifier, DEFAULT_NL_RULES[node.quantifier])
            return tmpl.format(var=var_str, body=body_nl)

        # --- Boolean ---
        if isinstance(node, BooleanNode):
            op = node.operator

            # Double-negation elimination: ¬¬A → A
            if (
                op == 'not'
                and isinstance(node.children[0], BooleanNode)
                and node.children[0].operator == 'not'
            ):
                return self._nl_node(
                    node.children[0].children[0],
                    rules, meanings, negative_meanings,
                    parenthesis, split_conj, parent_op,
                )

            # ¬(t₁ = t₂) — no registered negative meaning for '='
            if (
                op == 'not'
                and isinstance(node.children[0], RelationNode)
                and node.children[0].name == '='
                and '=' not in negative_meanings
            ):
                rel = node.children[0]
                arg_nl = [meanings.get(a, a) for a in rel.arguments]
                return f"{arg_nl[0]} is not equal to {arg_nl[1]}"

            # ¬(t₁ < t₂) / ¬(t₁ > t₂) — no registered negative meaning
            if (
                op == 'not'
                and isinstance(node.children[0], RelationNode)
                and node.children[0].name in ('<', '>')
                and node.children[0].name not in negative_meanings
            ):
                rel = node.children[0]
                sym = 'less than' if rel.name == '<' else 'greater than'
                arg_nl = [meanings.get(a, a) for a in rel.arguments]
                return f"{arg_nl[0]} is not {sym} {arg_nl[1]}"

            # ¬R(…) with a registered negative meaning for R
            if (
                op == 'not'
                and isinstance(node.children[0], RelationNode)
                and node.children[0].name in negative_meanings
            ):
                rel = node.children[0]
                tmpl = negative_meanings[rel.name]
                arg_nl = [meanings.get(a, a) for a in rel.arguments]
                try:
                    return tmpl.format(*arg_nl)
                except (IndexError, KeyError):
                    return f"{tmpl}({', '.join(arg_nl)})"

            # Top-level conjunction splitting
            if op == 'and' and split_conj:
                left_nl = self._nl_node(
                    node.children[0], rules, meanings, negative_meanings,
                    parenthesis, split_conj=True, parent_op=op,
                )
                right_nl = self._nl_node(
                    node.children[1], rules, meanings, negative_meanings,
                    parenthesis, split_conj=True, parent_op=op,
                )
                right_nl = right_nl[0].upper() + right_nl[1:] if right_nl else right_nl
                return f"{left_nl}. {right_nl}"

            # General case: recurse
            children_nl = [
                self._nl_node(
                    c, rules, meanings, negative_meanings,
                    parenthesis, split_conj=False, parent_op=op,
                )
                for c in node.children
            ]

            # N-ary and/or formatting
            if op in ('and', 'or') and len(children_nl) > 2:
                join_word = ' and ' if op == 'and' else ' or '
                result = ', '.join(children_nl[:-1]) + f',{join_word}{children_nl[-1]}'
            else:
                tmpl = rules.get(op, DEFAULT_NL_RULES.get(op, op))
                try:
                    result = tmpl.format(*children_nl)
                except (IndexError, KeyError):
                    result = f' {op} '.join(children_nl)

            # Precedence-aware parenthesisation
            needs_parens = False
            if parenthesis and op in _BINARY_OPS:
                needs_parens = True
            elif (
                parent_op is not None
                and op in _NL_PREC
                and parent_op in _NL_PREC
                and _NL_PREC[op] < _NL_PREC[parent_op]
            ):
                needs_parens = True

            if needs_parens:
                result = f"({result})"
            return result

        # --- Relation ---
        if isinstance(node, RelationNode):
            # Built-in infix predicates with no user-provided meaning
            if node.name == '=' and '=' not in meanings and len(node.arguments) == 2:
                arg_nl = [meanings.get(a, a) for a in node.arguments]
                return f"{arg_nl[0]} equals {arg_nl[1]}"
            if node.name == '<' and '<' not in meanings and len(node.arguments) == 2:
                arg_nl = [meanings.get(a, a) for a in node.arguments]
                return f"{arg_nl[0]} is less than {arg_nl[1]}"
            if node.name == '>' and '>' not in meanings and len(node.arguments) == 2:
                arg_nl = [meanings.get(a, a) for a in node.arguments]
                return f"{arg_nl[0]} is greater than {arg_nl[1]}"

            tmpl = meanings.get(node.name, node.name)
            if node.arguments:
                arg_nl = [meanings.get(a, a) for a in node.arguments]
                try:
                    return tmpl.format(*arg_nl)
                except (IndexError, KeyError):
                    return f"{tmpl}({', '.join(arg_nl)})"
            return tmpl

        raise ValueError(f"Unknown node type: {type(node).__name__}")
