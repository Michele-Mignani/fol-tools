"""
fol_tools.formula
=================

High-level façade for working with a single FOL formula.

:class:`FOL` is the primary entry point for the library.  It wraps a raw
formula string and lazily computes the AST and signature on first access.
All heavy operations (parsing, Z3 encoding, solving) are deferred until
actually needed.

Lazy properties
---------------
Both :attr:`FOL.tree` and :attr:`FOL.signature` are computed on first
access and cached thereafter.  This means creating many ``FOL`` objects
is cheap; parsing only happens when you actually use them.

Signature override
------------------
You can pass a pre-built *signature* dict at construction time.  This is
useful when you know the domain vocabulary (e.g. which lowercase names
are constants vs. variables) and want to avoid relying on the
auto-extracted signature.  If provided, the user signature is returned
directly by :attr:`FOL.signature` without running the extractor.

Usage
-----
.. code-block:: python

    from fol_tools.formula import FOL
    from fol_tools.solver  import FOLSolver

    f1 = FOL("∀x (Human(x) → Mortal(x))")
    f2 = FOL("∀x (¬Mortal(x) → ¬Human(x))")

    print(f1.tree)                # AST root
    print(f1.signature)           # {'Rel': {'Human': 1, 'Mortal': 1}, 'Var': {'x'}, 'Const': set()}
    print(f1.to_smtlib())         # SMT-LIB string

    FOLSolver().are_equivalent(f1, f2)   # True

Forbidden operators
-------------------
The parser rejects ASCII operator sequences ``->`` and ``&``.
:meth:`validate` also checks for these before attempting to parse.
"""

from __future__ import annotations

from .ast import Node, QuantifierNode, BooleanNode, RelationNode, BoolConstNode
from .signature import _VAR_PATTERN


class FOL:
    """A First-Order Logic formula with lazy parsing and caching.

    Parameters
    ----------
    formula_str : str
        The Unicode FOL formula string.  Forbidden ASCII operators
        (``->``, ``&``) will cause :meth:`validate` to return ``False``
        and :attr:`tree` to raise ``ValueError``.
    signature : dict, optional
        A pre-built signature dict
        ``{'Rel': {…}, 'Var': {…}, 'Const': {…}}``.
        If provided, it overrides the auto-extracted signature and is
        returned verbatim by :attr:`signature`.  Useful when you know
        that certain lowercase names are domain constants, not variables.

    Attributes
    ----------
    formula_str : str
        The original formula string (never modified).
    """

    # ASCII sequences that are never valid in a well-formed formula
    _FORBIDDEN: frozenset[str] = frozenset({'->',  '&'})

    def __init__(self, formula_str: str, signature: dict | None = None) -> None:
        self.formula_str: str = formula_str
        self._tree: Node | None = None
        self._signature: dict | None = None
        self._user_signature: dict | None = signature

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @property
    def tree(self) -> Node:
        """Parse the formula string and return the root AST node (cached).

        The parse result is cached on first access; subsequent accesses
        return the same object without re-parsing.

        Returns
        -------
        Node
            Root of the AST.

        Raises
        ------
        ValueError
            If the formula contains syntax errors.
        """
        if self._tree is None:
            from .parser import FOLParser
            self._tree = FOLParser().parse(self.formula_str)
        return self._tree

    @property
    def signature(self) -> dict:
        """Return the logical signature of the formula (cached).

        If a user-provided signature was given at construction time, it is
        returned as-is without running the extractor.  Otherwise the
        signature is extracted from the AST on first access and cached.

        Returns
        -------
        dict
            ``{'Rel': {name: arity}, 'Var': {…}, 'Const': {…}}``.
        """
        if self._user_signature is not None:
            return self._user_signature
        if self._signature is None:
            from .signature import FOLSignature
            self._signature = FOLSignature().extract(self.tree)
        return self._signature

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Return ``True`` iff the formula is well-formed.

        Checks (in order):

        1. No forbidden ASCII operator sequences (``->``, ``&``).
        2. The formula is parseable by :class:`~fol_tools.parser.FOLParser`.
        3. The formula contains no free variables (every lowercase name
           that is not a known constant is bound by a quantifier).

        Returns
        -------
        bool
            ``True`` if all checks pass, ``False`` otherwise.  Exceptions
            during parsing are caught and mapped to ``False``.

        Notes
        -----
        This method deliberately does not raise on invalid input so that
        it can be used as a simple predicate in filtering pipelines.
        """
        try:
            for seq in self._FORBIDDEN:
                if seq in self.formula_str:
                    return False
            _ = self.tree           # triggers parse; raises on syntax error
            return len(self.free_variables()) == 0
        except Exception:
            return False

    def free_variables(self) -> set:
        """Return the set of free (unbound) variable names.

        A term is classified as a *variable candidate* iff it matches the
        pattern ``[a-z][0-9]*`` (single lowercase letter, optionally followed
        by digits — e.g. ``x``, ``y``, ``x1``).  Any other token is treated
        as a constant regardless of case (``tom``, ``diamondMine``, ``John``,
        ``42`` are all constants).

        A variable candidate is *free* if it is not bound by an enclosing
        quantifier and not listed in the user-provided signature's ``Const``
        set.

        Returns
        -------
        set[str]
            Set of free variable name strings (may be empty).
        """
        user_consts: set = self._user_signature['Const'] if self._user_signature else set()
        return self._collect_free(self.tree, bound=frozenset(), user_consts=user_consts)

    def bound_variables(self) -> set:
        """Return the set of all quantifier-bound variable names.

        Returns
        -------
        set[str]
            Every variable name that appears in the scope of a ``∀`` or
            ``∃`` quantifier in the formula.
        """
        return self._collect_bound(self.tree)

    def to_smtlib(self) -> str:
        """Encode the formula as an SMT-LIB 2 string.

        Uses Z3's ``sexpr()`` method on the encoded Z3 expression.

        Returns
        -------
        str
            SMT-LIB 2 formula string.

        Raises
        ------
        ValueError
            If the formula fails to parse or encode.
        """
        from .encoder import Z3ContextBuilder, FOLZ3Encoder
        ctx = Z3ContextBuilder(self.signature)
        symbols = ctx.build_symbols()
        encoder = FOLZ3Encoder(sort=ctx.sort)
        z3_expr = encoder.encode(self.tree, symbols)
        return z3_expr.sexpr()

    def to_string(self) -> str:
        """Re-serialise the formula to its canonical Unicode FOL form.

        Uses :class:`~fol_tools.translator.FOLTranslator` internally.

        Returns
        -------
        str
            Canonical Unicode formula string.
        """
        from .translator import FOLTranslator
        return FOLTranslator().to_string(self.tree)

    def to_nl(
        self,
        rules: dict | None = None,
        symbol_meanings: dict | None = None,
        negative_meanings: dict | None = None,
        parenthesis: bool = False,
        split_conjunctions: bool = False,
    ) -> str:
        """Generate a natural-language sentence for this formula.

        Delegates to :meth:`~fol_tools.translator.FOLTranslator.to_nl`.
        See that method for full parameter documentation.

        Returns
        -------
        str
            English sentence (capitalised, ending with a period).
        """
        from .translator import FOLTranslator
        return FOLTranslator().to_nl(
            self.tree,
            rules=rules,
            symbol_meanings=symbol_meanings,
            negative_meanings=negative_meanings,
            parenthesis=parenthesis,
            split_conjunctions=split_conjunctions,
        )

    def perturbations(self, n: int = 5) -> list[str]:
        """Return up to *n* rule-based structural perturbations.

        Delegates to :func:`~fol_tools.perturbations.generate_rule_based`.

        Parameters
        ----------
        n : int
            Maximum number of perturbations.

        Returns
        -------
        list[str]
            Distinct perturbation formula strings (length ≤ *n*).
        """
        from .perturbations import generate_rule_based
        return generate_rule_based(self.formula_str, n)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"FOL({self.formula_str!r})"

    def __str__(self) -> str:  # pragma: no cover
        return self.formula_str

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_free(self, node: Node, bound: frozenset, user_consts: set) -> set:
        """Recursively collect all free variable names in *node*."""
        if isinstance(node, BoolConstNode):
            return set()
        if isinstance(node, QuantifierNode):
            return self._collect_free(node.body, bound | {node.variable}, user_consts)
        if isinstance(node, BooleanNode):
            result: set = set()
            for child in node.children:
                result |= self._collect_free(child, bound, user_consts)
            return result
        if isinstance(node, RelationNode):
            return {
                arg for arg in node.arguments
                if (
                    _VAR_PATTERN.match(arg)
                    and arg not in bound
                    and arg not in user_consts
                )
            }
        return set()

    def _collect_bound(self, node: Node) -> set:
        """Recursively collect all quantifier-bound variable names in *node*."""
        if isinstance(node, BoolConstNode):
            return set()
        if isinstance(node, QuantifierNode):
            return {node.variable} | self._collect_bound(node.body)
        if isinstance(node, BooleanNode):
            result: set = set()
            for child in node.children:
                result |= self._collect_bound(child)
            return result
        return set()
