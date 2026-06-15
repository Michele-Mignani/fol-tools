"""
fol_tools.parser
================

Recursive-descent parser for the Unicode FOL formula language.

Accepted formula syntax
-----------------------
All logical symbols must be written as Unicode characters; ASCII
alternatives (``->``, ``&``) are **rejected** to prevent silent bugs.

Operators (in order of increasing precedence, lowest binds last):

    ↔  biconditional          (left-associative)
    ⊕  exclusive-or           (left-associative)
    →  implication            (right-associative)
    ∨  disjunction            (left-associative)
    ∧  conjunction            (left-associative)
    ¬  negation               (prefix, right-associative)
    ∀∃ quantifiers            (limited scope — see below)
    () parenthesised sub-expr (highest precedence)

Quantifier scope
~~~~~~~~~~~~~~~~
Quantifiers bind with *limited scope*: they apply only to the smallest
primary expression immediately following them, not to the entire
remainder of the formula.

.. code-block:: text

    ∃y(P(y)) → Q(y)   parses as   (∃y(P(y))) → Q(y)
    ∀x ¬P(x)          parses as   ∀x(¬P(x))
    ∀x ∃y P(x,y)      parses as   ∀x(∃y(P(x,y)))   ← chained primaries

This matches the scoping used in the FOLIO and MALLS datasets.

Relation atoms
~~~~~~~~~~~~~~
A **relation** is an uppercase-initial identifier, optionally followed by
a parenthesised argument list.  A zero-argument relation is a propositional
atom.

A **term** is any identifier (upper- or lower-case) or a numeric token.

Equality / inequality
~~~~~~~~~~~~~~~~~~~~~
``t₁ = t₂`` and ``t₁ ≠ t₂`` are both accepted.  The ``≠`` form desugars
to ``¬(t₁ = t₂)``.

Boolean constants
~~~~~~~~~~~~~~~~~
``⊤`` / ``True`` → tautology,  ``⊥`` / ``False`` → contradiction.

Usage example
-------------
.. code-block:: python

    from fol_tools.parser import FOLParser

    parser = FOLParser()
    tree = parser.parse("∀x (Human(x) → Mortal(x))")

Extending the parser
--------------------
To add a new operator at precedence level *k*:

1. Add its Unicode character to ``_OPERATORS``.
2. Insert a new ``_parse_<name>`` method at the right nesting level
   (between the two adjacent existing levels).
3. Call it from the level above and call the level below from within it.
"""

from __future__ import annotations

from .ast import Node, QuantifierNode, BooleanNode, RelationNode, BoolConstNode


class FOLParser:
    """Parse a Unicode FOL formula string into an AST.

    The parser is **not thread-safe**: it stores per-parse state in
    ``self._tokens`` and ``self._pos``.  Instantiate one parser per
    thread, or instantiate a new one per call (they are lightweight).

    Methods
    -------
    parse(formula_str)
        The only public method.  Returns the root :class:`~fol_tools.ast.Node`.

    Raises
    ------
    ValueError
        On any syntax error: forbidden ASCII operators, unexpected tokens,
        unbalanced parentheses, lowercase relation names, etc.
    """

    # Single-character Unicode tokens.  Everything else is an identifier
    # (alphanumeric / underscore / hyphen) or a numeric literal.
    _OPERATORS: frozenset[str] = frozenset('∀∃¬∧∨→↔⊕()=≠⊤⊥')

    # Mapping from surface token to bool value for Boolean constants.
    _BOOL_CONSTS: dict[str, bool] = {'True': True, 'False': False, '⊤': True, '⊥': False}

    # ASCII operator sequences that must never appear in a valid formula.
    _FORBIDDEN: frozenset[str] = frozenset({'->', '&'})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, formula_str: str) -> Node:
        """Parse *formula_str* and return the root AST node.

        Parameters
        ----------
        formula_str : str
            A Unicode FOL formula.  Leading/trailing whitespace is ignored.
            Commas inside argument lists are treated as whitespace.

        Returns
        -------
        Node
            Root of the abstract syntax tree.

        Raises
        ------
        ValueError
            If the formula contains forbidden ASCII operators, an
            unexpected character, an imbalanced parenthesis, a lowercase
            relation name, or any other syntax error.
        """
        for seq in self._FORBIDDEN:
            if seq in formula_str:
                raise ValueError(
                    f"Forbidden operator {seq!r} in formula; "
                    "use Unicode symbols (→ instead of ->, ∧ instead of &)"
                )

        self._tokens: list[str] = self._tokenize(formula_str)
        self._pos: int = 0

        if not self._tokens:
            raise ValueError("Formula string is empty")

        result = self._parse_formula()

        if self._pos != len(self._tokens):
            raise ValueError(
                f"Unexpected token {self._tokens[self._pos]!r} "
                f"at position {self._pos} — formula was not fully consumed"
            )
        return result

    # ------------------------------------------------------------------
    # Tokeniser
    # ------------------------------------------------------------------

    def _tokenize(self, s: str) -> list[str]:
        """Split *s* into a flat list of token strings.

        Token kinds
        -----------
        * Single-character operator from ``_OPERATORS``.
        * Alphanumeric-plus-hyphen identifier (hyphens are allowed
          *inside* identifiers, not at the start or end).
        * Digit-starting numeric literal (``[0-9][0-9a-zA-Z]*``).
        * Whitespace and commas are separators and produce no tokens.

        Raises
        ------
        ValueError
            On any character that does not belong to the above categories.
        """
        tokens: list[str] = []
        i = 0
        while i < len(s):
            ch = s[i]

            # Whitespace and commas are separators
            if ch.isspace() or ch == ',':
                i += 1
                continue

            # Single-char operator token
            if ch in self._OPERATORS:
                tokens.append(ch)
                i += 1
                continue

            # Alphabetic / underscore: start of an identifier
            if ch.isalpha() or ch == '_':
                j = i
                while j < len(s) and (
                    s[j].isalnum() or s[j] == '_'
                    or (
                        s[j] == '-'
                        and j + 1 < len(s)
                        and (s[j + 1].isalnum() or s[j + 1] == '_')
                    )
                ):
                    j += 1
                tokens.append(s[i:j])
                i = j
                continue

            # Digit: start of a numeric literal
            if ch.isdigit():
                j = i
                while j < len(s) and s[j].isalnum():
                    j += 1
                tokens.append(s[i:j])
                i = j
                continue

            raise ValueError(
                f"Unexpected character {ch!r} (U+{ord(ch):04X}) in formula"
            )
        return tokens

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _current(self) -> str | None:
        """Return the current token without consuming it, or None at EOF."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self, expected: str | None = None) -> str:
        """Consume and return the current token.

        Parameters
        ----------
        expected : str, optional
            If given, raise ``ValueError`` if the current token differs.

        Raises
        ------
        ValueError
            If *expected* does not match, or if we are already at EOF.
        """
        tok = self._current()
        if expected is not None and tok != expected:
            raise ValueError(
                f"Expected {expected!r} but got {tok!r} "
                f"at position {self._pos}"
            )
        if tok is None:
            raise ValueError("Unexpected end of formula")
        self._pos += 1
        return tok

    # ------------------------------------------------------------------
    # Grammar rules (top-down, precedence encoded by nesting depth)
    #
    # Precedence ladder (outermost → innermost, i.e. lowest → highest):
    #   _parse_formula  (entry)
    #     _parse_biconditional  ↔
    #       _parse_xor          ⊕
    #         _parse_implication  →
    #           _parse_disjunction  ∨
    #             _parse_conjunction  ∧
    #               _parse_negation  ¬
    #                 _parse_primary  ∀∃ () atoms
    # ------------------------------------------------------------------

    def _parse_formula(self) -> Node:
        """Entry point; delegates to the lowest-precedence level."""
        return self._parse_biconditional()

    # --- Biconditional (↔) — left-associative, precedence 1 ---

    def _parse_biconditional(self) -> Node:
        """biconditional → xor (↔ xor)*"""
        left = self._parse_xor()
        while self._current() == '↔':
            self._consume('↔')
            right = self._parse_xor()
            left = BooleanNode('iff', [left, right])
        return left

    # --- Exclusive-or (⊕) — left-associative, precedence 2 ---

    def _parse_xor(self) -> Node:
        """xor → implication (⊕ implication)*"""
        left = self._parse_implication()
        while self._current() == '⊕':
            self._consume('⊕')
            right = self._parse_implication()
            left = BooleanNode('xor', [left, right])
        return left

    # --- Implication (→) — right-associative, precedence 3 ---

    def _parse_implication(self) -> Node:
        """implication → disjunction (→ disjunction)?  [right-recursive]"""
        left = self._parse_disjunction()
        if self._current() == '→':
            self._consume('→')
            right = self._parse_implication()   # right-recursive → right-assoc
            return BooleanNode('implies', [left, right])
        return left

    # --- Disjunction (∨) — left-associative, precedence 4 ---

    def _parse_disjunction(self) -> Node:
        """disjunction → conjunction (∨ conjunction)*"""
        left = self._parse_conjunction()
        while self._current() == '∨':
            self._consume('∨')
            right = self._parse_conjunction()
            left = BooleanNode('or', [left, right])
        return left

    # --- Conjunction (∧) — left-associative, precedence 5 ---

    def _parse_conjunction(self) -> Node:
        """conjunction → negation (∧ negation)*"""
        left = self._parse_negation()
        while self._current() == '∧':
            self._consume('∧')
            right = self._parse_negation()
            left = BooleanNode('and', [left, right])
        return left

    # --- Negation (¬) — prefix, right-associative, precedence 6 ---

    def _parse_negation(self) -> Node:
        """negation → ¬ negation | primary"""
        if self._current() == '¬':
            self._consume('¬')
            operand = self._parse_negation()    # right-recursive ⇒ ¬¬A ok
            return BooleanNode('not', [operand])
        return self._parse_primary()

    # --- Primary — quantifiers, parentheses, atoms ---

    def _parse_primary(self) -> Node:
        """Parse the highest-precedence expression.

        Handles:

        * Quantifiers (``∀`` / ``∃``) — *limited scope*: the quantifier
          applies to the next ``_parse_negation()`` call only, not to the
          full formula remainder.
        * Parenthesised sub-formulas ``( formula )``.
        * Boolean constants (``⊤``, ``⊥``, ``True``, ``False``).
        * Equality atoms ``term = term`` and inequality ``term ≠ term``.
        * Relation atoms ``RelName(args…)`` or zero-arity ``RelName``.
        """
        tok = self._current()

        # --- Quantifier (limited scope) ---
        if tok in ('∀', '∃'):
            return self._parse_quantifier_primary()

        # --- Parenthesised sub-formula ---
        if tok == '(':
            self._consume('(')
            result = self._parse_formula()
            self._consume(')')
            return result

        # --- Boolean constants ---
        if tok in self._BOOL_CONSTS:
            self._consume()
            return BoolConstNode(self._BOOL_CONSTS[tok])

        # --- Identifier or numeric token: relation atom or equality ---
        if tok and (tok[0].isalpha() or tok[0] == '_' or tok[0].isdigit()):
            name = self._consume()

            # Equality: term '=' term
            if self._current() == '=':
                self._consume('=')
                rhs = self._current()
                if not rhs or not (rhs[0].isalpha() or rhs[0] == '_' or rhs[0].isdigit()):
                    raise ValueError(
                        f"Expected a term after '=', got {rhs!r}"
                    )
                self._consume()
                return RelationNode('=', [name, rhs])

            # Inequality: term '≠' term  →  ¬(term = term)
            if self._current() == '≠':
                self._consume('≠')
                rhs = self._current()
                if not rhs or not (rhs[0].isalpha() or rhs[0] == '_' or rhs[0].isdigit()):
                    raise ValueError(
                        f"Expected a term after '≠', got {rhs!r}"
                    )
                self._consume()
                return BooleanNode('not', [RelationNode('=', [name, rhs])])

            # Lowercase / underscore token that is NOT followed by '=' is invalid
            # (variables may only appear as *arguments* to a relation, not standalone)
            if name[0].islower() or name[0] == '_':
                raise ValueError(
                    f"Unexpected lowercase token {name!r}: "
                    "expected a relation name (uppercase) or 'term = term'"
                )

            # Numeric token standing alone is also invalid
            if name[0].isdigit():
                raise ValueError(
                    f"Unexpected numeric token {name!r}: "
                    "numeric literals may only appear as relation arguments"
                )

            return self._parse_relation_body(name)

        raise ValueError(
            f"Unexpected token {tok!r} — "
            "expected a quantifier, '(', Boolean constant, or relation name"
        )

    def _parse_quantifier_primary(self) -> Node:
        """Parse a quantifier with *limited scope*.

        The quantifier binds to the immediately following
        ``_parse_negation()`` result only.  This gives the correct
        behaviour for chained quantifiers (``∀x∃y P(x,y)``) and prevents
        a quantifier from accidentally swallowing a subsequent binary
        operator (``∃y(P(y)) → Q(y)`` stays ``(∃y P(y)) → Q(y)``).

        Grammar::

            quantifier → ('∀'|'∃') variable negation_expr
        """
        q_tok = self._consume()
        quantifier = 'forall' if q_tok == '∀' else 'exists'

        variable = self._consume()
        if not variable or not variable[0].isalpha() or variable[0].isupper():
            raise ValueError(
                f"Expected a lowercase variable name after the quantifier, "
                f"got {variable!r}"
            )

        body = self._parse_negation()
        return QuantifierNode(quantifier, variable, body)

    def _parse_relation_body(self, name: str) -> Node:
        """Parse the optional argument list for an already-consumed relation name.

        If the next token is ``(``, consume the argument list.  Otherwise
        return a zero-arity :class:`~fol_tools.ast.RelationNode`.

        Parameters
        ----------
        name : str
            Relation name (already consumed by the caller).
        """
        if self._current() == '(':
            self._consume('(')
            args: list[str] = []
            while self._current() != ')':
                tok = self._current()
                if tok is None:
                    raise ValueError(
                        f"Unexpected end of formula inside argument list of {name!r}"
                    )
                args.append(self._consume())
            self._consume(')')
            return RelationNode(name, args)
        return RelationNode(name, [])
