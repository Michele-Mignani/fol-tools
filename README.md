# fol_tools

A Python library for working with **First-Order Logic (FOL)** formulas
written in Unicode syntax.

## Features

| Module | What it does |
|--------|-------------|
| `parser` | Parse a Unicode formula string into an AST |
| `signature` | Extract / merge the non-logical symbol catalogue |
| `encoder` | Convert an AST into a Z3 expression |
| `solver` | Check satisfiability, equivalence, and entailment via Z3 |
| `translator` | Serialise an AST back to Unicode or natural language |
| `perturbations` | Generate near-miss formula variants |
| `formula` | High-level `FOL` façade with lazy parsing and caching |

## Formula syntax

All logical symbols must be **Unicode** (ASCII alternatives `->` and `&` are rejected).

| Symbol | Meaning | Precedence (low → high) |
|--------|---------|------------------------|
| `↔` | biconditional | 1 (binds last) |
| `⊕` | exclusive or | 2 |
| `→` | implication (right-assoc) | 3 |
| `∨` | disjunction | 4 |
| `∧` | conjunction | 5 |
| `¬` | negation | 6 |
| `∀` `∃` | quantifiers (limited scope) | 7 (tightest) |

Relations start with an uppercase letter; variables are lowercase;
constants are uppercase or digit-starting.

```
∀x (Human(x) → Mortal(x))
∃x ∃y (Loves(x, y) ∧ x ≠ y)
∀x ∀y (x = y ↔ (A(x) ↔ A(y)))
```

## Installation

```bash
pip install -e ".[dev]"   # editable install with test deps
```

Requires Python ≥ 3.10 and `z3-solver`.

## Quick start

```python
from fol_tools import FOL, FOLSolver, FOLTranslator

# --- Parsing and inspection ---
f = FOL("∀x (Human(x) → Mortal(x))")
print(f.validate())          # True
print(f.signature)           # {'Rel': {'Human': 1, 'Mortal': 1}, 'Var': {'x'}, 'Const': set()}
print(f.to_smtlib())         # (forall ((x U)) (=> (Human x) (Mortal x)))

# --- Equivalence checking ---
solver = FOLSolver()
f2 = FOL("∀x (¬Mortal(x) → ¬Human(x))")
print(solver.are_equivalent(f, f2))   # True  — contrapositive

# --- Natural language ---
print(f.to_nl(symbol_meanings={'Human': '{0} is human', 'Mortal': '{0} is mortal'}))
# For all x, if x is human, then x is mortal.

# --- Perturbations ---
for v in f.perturbations(n=4):
    print(v)
```

## Running the tests

```bash
cd /path/to/fol_tools
pytest -v
```

## Module overview

```
fol_tools/
├── ast.py           Node, QuantifierNode, BooleanNode, RelationNode, BoolConstNode
├── parser.py        FOLParser          — string → AST
├── signature.py     FOLSignature       — extract / merge symbol catalogue
├── encoder.py       Z3ContextBuilder, FOLZ3Encoder  — AST → Z3 expression
├── solver.py        FOLSolver, SolverTimeoutError   — satisfiability / equivalence / entailment
├── translator.py    FOLTranslator      — AST → string / tree string / NL
├── perturbations.py generate_rule_based, RULE_TRANSFORMS
├── formula.py       FOL                — high-level façade
└── __init__.py      public re-exports
```

## Extending the library

* **New operator**: add its Unicode char to `FOLParser._OPERATORS`, insert a
  `_parse_<name>` method at the right precedence level, and handle it in
  `FOLZ3Encoder._encode_boolean` and `FOLTranslator`.

* **New AST node kind**: subclass `Node` in `ast.py`, then update every
  visitor (encoder, translator, signature, perturbations, formula).

* **New perturbation**: write a `transform(node) -> Node | None` function
  and append it to `perturbations.RULE_TRANSFORMS`.

* **Multiple sorts**: replace the single `DeclareSort("U")` in
  `Z3ContextBuilder` with a sort-inference step; the encoder needs no
  changes as long as `symbols[name]` resolves to a correctly-typed Z3 object.
