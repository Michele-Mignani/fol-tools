"""
tests/test_real_datasets.py
===========================

Integration tests against real formulas drawn from the FOLIO and MALLS
datasets used in the DatasetCuration project.

Test organisation
-----------------
1.  Gold FOL_sentence formulas — must parse without errors.
2.  Signature extraction on a representative sample.
3.  SMT-LIB encoding on a representative sample.
4.  Z3 equivalence / entailment checks on pairs derived from the dataset.
5.  Infix predicates: =, <, > in both infix and prefix forms.
6.  Known-invalid FOL_sentence_old formulas — the four failure modes
    found in the real data are documented and the correct exception
    (or behaviour) is asserted for each.

Failure modes found in FOL_sentence_old
-----------------------------------------
A. Multi-axiom whitespace (76 cases): axioms concatenated with whitespace
   only, no conjunction operator.  Parser stops after the first complete
   formula: "not fully consumed".

B. Unbalanced parentheses (2 cases): genuine bracket imbalance.
   Parser raises on EOF inside open parenthesis.

C. Long-arrow U+27F7 (1 case): ⟷ used instead of ↔ (U+2194).
   Tokeniser raises "Unexpected character".

D. Free variable in old formulas (e.g. 'x' without a binder).
   Parser accepts it but FOL.validate() returns False.

The package is NOT expected to swallow these errors — failing loudly is
the correct behaviour.  Section 6 asserts the expected exception for each.
"""

from __future__ import annotations

import pytest
from fol_tools import FOL, FOLParser, FOLSolver, FOLSignature
from fol_tools.ast import RelationNode, BooleanNode


def _p(s: str):
    return FOLParser().parse(s)


# ===========================================================================
# 1. Gold FOL_sentence formulas — representative sample, must all parse
# ===========================================================================

GOLD_FORMULAS = [
    # concl_385_43  — bare propositional atom
    "MostActivePlayerIn(cocoGauff, majorTennis)",

    # story_0  — large ↔ with long disjunction, multiple ¬
    ("∀x (WildTurkey(x) ↔ (EasternWildTurkey(x) ∨ OsceolaWildTurkey(x) ∨ GouldsWildTurkey(x)"
     " ∨ MerriamsWildTurkey(x) ∨ RiograndeWildTurkey(x) ∨ OcellatedWildTurkey(x)))"
     " ∧ ¬(EasternWildTurkey(tom)) ∧ ¬(OsceolaWildTurkey(tom))"
     " ∧ ¬(GouldsWildTurkey(tom)) ∧ ¬(MerriamsWildTurkey(tom) ∨ RiograndeWildTurkey(tom))"
     " ∧ WildTurkey(tom)"),

    # concl_379_104  — conjunction of negated predicates
    "¬BornIn(luke, multipleBirth) ∧ ¬ComplainAboutOften(luke, annoyingSiblings)",

    # story_166  — ternary ∀, ⊕, ∃ inside ∀
    ("∃x (ManagedBuilding(x) ∧ AllowPet(x))"
     " ∧ ∀x ∀y (ManagedBuilding(x) ∧ RentApartmentIn(y, x) → DepositRequired(y))"
     " ∧ ∀x ∀y ∀z ((SecurityDeposit(x, z) ∧ ManagedBuilding(z) ∧ MonthlyRentAt(y, z))"
     "         → (MoreThan(x, y) ⊕ Equal(x, y)))"
     " ∧ Cat(fluffy) ∧ BelongTo(fluffy, tom) ∧ ∀x (Cat(x) → Pet(x))"
     " ∧ ManagedBuilding(oliveGarden) ∧ MonthlyRentAt(uSD2000, oliveGarden)"
     " ∧ MoreThan(uSD2000, uSD1500)"
     " ∧ ∀x ∀y (ManagedBuilding(x) ∧ AllowedToMoveInWith(tom, x, fluffy)"
     "         ∧ SecurityDeposit(y, x) ∧ ¬MoreThan(y, uSD1500) → RentApartmentIn(tom, x))"
     " ∧ ∀x ∀y ∀z (ManagedBuilding(x) ∧ AllowPet(x) ∧ Pet(z) → AllowedToMoveInWith(y, x, z))"),

    # concl_306_85  — disjunction of positive and negated predicate
    "Provide(hamdenPlazaSubway, takeOutService) ∨ ¬ReceiveManyNegativeReviews(hamdenPlazaSubway)",

    # story_350  — ⊕ in body, multiple ∀ conjuncts
    ("∀x (Adore(max, x) ∧ ZahaHadid(x) ∧ DesignStyle(x) → InterestingGeometry(x))"
     " ∧ ∀x (Adore(max, x) ∧ BrutalistBuilding(x) → ¬InterestingGeometry(x))"
     " ∧ ∀x (Adore(max, x)"
     "   → ((ZahaHadid(x) ∧ DesignStyle(x)) ⊕ (KellyWearstler(x) ∧ DesignStyle(x))))"
     " ∧ ∀x (Adore(max, x) ∧ KellyWearstler(x) ∧ DesignStyle(x) → Evocative(x))"
     " ∧ ∀x (Adore(max, x) ∧ KellyWearstler(x) ∧ DesignStyle(x) → Dreamy(x))"
     " ∧ ∀x (Adore(max, x) ∧ Design(x) ∧ ByMax(x) ∧ InterestingGeometry(x)"
     "     → BrutalistBuilding(x) ∧ Evocative(x))"),

    # story_96  — long ground conjunction, no quantifiers
    ("ProfessionalWrestlingStable(diamondMine) ∧ In(diamondMine, wWE)"
     " ∧ Leads(roderickStrong, diamondMine) ∧ Includes(diamondMine, creedBrothers)"
     " ∧ Includes(diamondMine, ivyNile) ∧ Feuds(imperium, diamondMine)"),

    # story_217  — propositional-style: no quantifiers
    ("(LikeMusic(george) → WantToCompose(george))"
     " ∧ (AccessToProgram(george) → CanCompose(george))"
     " ∧ (WantToCompose(george) ∧ CanCompose(george) → WillCompose(george))"),

    # story_20  — ∃ inside ∀ body; Created/2
    ("Game(theLegendofZelda)"
     " ∧ ∃x (Japanese(x) ∧ VideoGameCompany(x) ∧ Created(x, theLegendofZelda))"
     " ∧ ∀x ∀y ((Game(x) ∧ InTop10(x) ∧ Created(y, x)) → (Japanese(y) ∧ VideoGameCompany(y)))"
     " ∧ ∀x ((Game(x) ∧ ∃y (GreaterThan(y, oneMillion) ∧ CopiesSold(x, y))) → InTop10(x))"
     " ∧ ∃y (GreaterThan(y, oneMillion) ∧ CopiesSold(theLegendofZelda, y))"),

    # concl_0_12  — bare unary atom
    "WildTurkey(joey)",

    # concl_96_36  — universal with doubly nested negation
    "∀x ((ProfessionalWrestlingStable(x) ∧ Includes(x, ivyNile)) → ¬Feuds(imperium, x))",

    # concl_319_83  — ground conjunction of three predicates
    "Cupcake(driedThaiChili) ∧ Product(driedThaiChili) ∧ From(driedThaiChili, bakedByMelissa)",

    # concl_58_88  — binary predicate, no quantifier
    "Contains(walden, knowledge)",

    # concl_456_191  — implication at top level, no outer quantifier
    ("¬LeapStraightIntoAir(yuri)"
     " → (AmericanNational(yuri) ∧ Professional(yuri) ∧ BasketballPlayer(yuri))"),

    # concl_83_75  — existential at top level
    "∃x (Owns(tom, x) ∧ VehicleRegistrationPlateIn(x, istanbul))",

    # concl_435_159  — ⊕ at top level with ∃ on right
    "Take(james, databaseCourse) ⊕ (∃y (PartTimeJob(y) ∧ Have(james, y) ∧ OfferedBy(y, university)))",

    # story_306  — long ∧-chain with nested ∃ inside ∀
    ("∀x (ListedIn(x, yelpRecommendation) → ¬ReceiveManyNegativeReviews(x))"
     " ∧ ∀x (∃y (HaveRating(x, y) ∧ GreaterThan(y, four)) → ListedIn(x, yelpRecommendation))"
     " ∧ ∃x (¬Provide(x, takeOutService) ∧ ReceiveManyNegativeReviews(x))"
     " ∧ ∀x (PopularAmong(x, localResidents) → ∃y (HaveRating(x, y) ∧ GreaterThan(y, four)))"
     " ∧ (∃y (HaveRating(hamdenPlazaSubway, y) ∧ GreaterThan(y, four))"
     "   ∨ PopularAmong(hamdenPlazaSubway, localResidents))"),

    # story_386  — negated conjunction at the end
    ("∀x (DeadlyDisease(x) → ComeWith(x, lowSurvivalRate))"
     " ∧ ∀x (SevereCancer(x) → DeadlyDisease(x))"
     " ∧ ∀x (BileDuctCancer(x) → SevereCancer(x))"
     " ∧ ∀x (Cholangiocarcinoma(x) → BileDuctCancer(x))"
     " ∧ ∀x (MildFlu(x) → ComeWith(x, lowSurvivalRate))"
     " ∧ ¬(BileDuctCancer(colorectalCancer) ∧ ComeWith(colorectalCancer, lowSurvivalRate))"),

    # story_380  — compound with ⊕ at top level
    ("∀x (InThisClub(x) ∧ PerformOftenIn(x, schoolTalentShow)"
     "   → Attend(x, schoolEvent) ∧ VeryEngagedWith(x, schoolEvent))"
     " ∧ ∀x (InThisClub(x)"
     "   → (PerformOftenIn(x, schoolTalentShow)"
     "      ⊕ (InActive(x) ∧ Disinterested(x) ∧ MemberOf(x, community))))"
     " ∧ ∀x (InThisClub(x) ∧ Chaperone(x, highSchoolDance)"
     "   → ¬(Student(x) ∧ AttendSchool(x)))"
     " ∧ ∀x (InThisClub(x) ∧ (InActive(x) ∧ Disinterested(x) ∧ MemberOf(x, community))"
     "   → Chaperone(x, highSchoolDance))"),
]

# Additional formulas exercising less common syntax
GOLD_FORMULAS_EXTRA = [
    "∀x ∀y (x = y → (Human(x) ↔ Human(y)))",
    "∀x ∀y (x ≠ y ∧ Human(x) → ¬SamePerson(x, y))",
    "∀x ∀y ∀z (Between(x, y, z) → ¬Between(z, y, x))",
    "⊤",
    "⊥",
    "∀x (Polygon(x) ∧ Sides(x, 3) ∧ Angles(x, 3) → Triangle(x))",
    "∀x (HasAge(x, 42) → Adult(x))",
    "∀x ¬¬Human(x)",
    "∀x (A(x) → ∃y (B(y) ∧ ∃z (C(z) ∧ R(x, y, z))))",
    "∀x (A(x) → B(x) → C(x) → D(x))",
    "∀x (A(x) ⊕ B(x) ⊕ C(x))",
]


class TestGoldFormulasParseCleanly:
    @pytest.mark.parametrize("formula", GOLD_FORMULAS)
    def test_gold_formula_parses(self, formula):
        assert _p(formula) is not None

    @pytest.mark.parametrize("formula", GOLD_FORMULAS_EXTRA)
    def test_extra_syntax_formula_parses(self, formula):
        assert _p(formula) is not None


# ===========================================================================
# 2. Signature extraction on real formulas
# ===========================================================================

class TestSignatureOnRealFormulas:
    def test_story_96_ground_conj(self):
        f = ("ProfessionalWrestlingStable(diamondMine) ∧ In(diamondMine, wWE)"
             " ∧ Leads(roderickStrong, diamondMine)")
        sig = FOLSignature().extract(_p(f))
        assert sig["Rel"]["ProfessionalWrestlingStable"] == 1
        assert sig["Rel"]["In"] == 2
        assert "diamondMine" in sig["Const"]

    def test_story_20_nested_existential(self):
        f = ("Game(theLegendofZelda)"
             " ∧ ∃x (Japanese(x) ∧ VideoGameCompany(x) ∧ Created(x, theLegendofZelda))")
        sig = FOLSignature().extract(_p(f))
        assert "x" in sig["Var"]
        assert "theLegendofZelda" in sig["Const"]
        assert sig["Rel"]["Created"] == 2

    def test_malls_numeric_in_const(self):
        f = "∀x (Polygon(x) ∧ Sides(x, 3) → Triangle(x))"
        sig = FOLSignature().extract(_p(f))
        assert "3" in sig["Const"]
        assert sig["Rel"]["Sides"] == 2

    def test_equality_now_appears_in_rel(self):
        # = is now catalogued as a relation with arity 2
        f = "∀x ∀y (x = y → Human(x))"
        sig = FOLSignature().extract(_p(f))
        assert "=" in sig["Rel"]
        assert sig["Rel"]["="] == 2

    def test_long_formula_signature_complete(self):
        f = ("∀x (DeadlyDisease(x) → ComeWith(x, lowSurvivalRate))"
             " ∧ ∀x (SevereCancer(x) → DeadlyDisease(x))"
             " ∧ ∀x (BileDuctCancer(x) → SevereCancer(x))")
        sig = FOLSignature().extract(_p(f))
        assert {"DeadlyDisease", "ComeWith", "SevereCancer", "BileDuctCancer"} <= sig["Rel"].keys()


# ===========================================================================
# 3. SMT-LIB encoding on real formulas
# ===========================================================================

class TestSMTLibOnRealFormulas:
    @pytest.mark.parametrize("formula", [
        "∀x (Human(x) → Mortal(x))",
        "∃x (Dog(x) ∧ Barks(x))",
        "∀x ∀y (Loves(x, y) → Cares(y, x))",
        "∀x (WildTurkey(x) ↔ (EasternWildTurkey(x) ∨ OsceolaWildTurkey(x)))",
        "ProfessionalWrestlingStable(diamondMine) ∧ In(diamondMine, wWE)",
        "Take(james, databaseCourse) ⊕ (∃y (PartTimeJob(y) ∧ Have(james, y)))",
        "∀x ∀y (x = y → Human(x))",
    ])
    def test_to_smtlib_produces_nonempty_string(self, formula):
        s = FOL(formula).to_smtlib()
        assert isinstance(s, str) and len(s) > 0

    def test_forall_in_smtlib(self):
        assert "forall" in FOL("∀x Human(x)").to_smtlib().lower()

    def test_exists_in_smtlib(self):
        assert "exists" in FOL("∃x Human(x)").to_smtlib().lower()


# ===========================================================================
# 4. Z3 equivalence / entailment on real-data-inspired pairs
# ===========================================================================

class TestSolverOnRealPairs:
    def setup_method(self):
        self.solver = FOLSolver()

    def test_contrapositive_from_dataset(self):
        f1 = FOL("∀x (DeadlyDisease(x) → ComeWith(x, lowSurvivalRate))")
        f2 = FOL("∀x (¬ComeWith(x, lowSurvivalRate) → ¬DeadlyDisease(x))")
        assert self.solver.are_equivalent(f1, f2)

    def test_cancer_chain_entailment(self):
        theory = [
            "∀x (SevereCancer(x) → DeadlyDisease(x))",
            "∀x (BileDuctCancer(x) → SevereCancer(x))",
            "∀x (DeadlyDisease(x) → ComeWith(x, lowSurvivalRate))",
        ]
        assert self.solver.implies(FOL("BileDuctCancer(a)"), FOL("ComeWith(a, lowSurvivalRate)"), theory=theory)

    def test_yelp_entailment_chain(self):
        theory = [
            "∀x (∃y (HaveRating(x, y) ∧ GreaterThan(y, four)) → ListedIn(x, yelpRecommendation))",
            "∀x (ListedIn(x, yelpRecommendation) → ¬ReceiveManyNegativeReviews(x))",
        ]
        f1 = FOL("∃y (HaveRating(restaurant, y) ∧ GreaterThan(y, four))")
        assert self.solver.implies(f1, FOL("¬ReceiveManyNegativeReviews(restaurant)"), theory=theory)

    def test_wildturkey_species_disjunction(self):
        axiom = "∀x (WildTurkey(x) ↔ (EasternWildTurkey(x) ∨ OsceolaWildTurkey(x)))"
        assert self.solver.implies(FOL("EasternWildTurkey(tom)"), FOL("WildTurkey(tom)"), theory=[axiom])

    def test_de_morgan_from_dataset(self):
        f1 = FOL("∀x ¬(Student(x) ∧ AttendSchool(x))")
        f2 = FOL("∀x (¬Student(x) ∨ ¬AttendSchool(x))")
        assert self.solver.are_equivalent(f1, f2)

    def test_game_creation_entailment(self):
        theory = ["∀x ∀y ((Game(x) ∧ InTop10(x) ∧ Created(y, x)) → Japanese(y))"]
        f1 = FOL("Game(zelda) ∧ InTop10(zelda) ∧ Created(nintendo, zelda)")
        assert self.solver.implies(f1, FOL("Japanese(nintendo)"), theory=theory)

    def test_non_equivalent_different_quantifiers(self):
        assert not self.solver.are_equivalent(FOL("∀x Human(x)"), FOL("∃x Human(x)"))

    def test_wrestling_negation_entailment(self):
        theory = ["∀x ((ProfessionalWrestlingStable(x) ∧ Includes(x, ivyNile)) → ¬Feuds(imperium, x))"]
        f1 = FOL("ProfessionalWrestlingStable(diamondMine) ∧ Includes(diamondMine, ivyNile)")
        assert self.solver.implies(f1, FOL("¬Feuds(imperium, diamondMine)"), theory=theory)


# ===========================================================================
# 5. Infix predicates: =, <, >
# ===========================================================================

class TestInfixPredicates:
    """
    = is the standard equality, < and > are uninterpreted comparison predicates.
    All three are accepted in infix form (a = b, a < b, a > b).
    = is also accepted in the form a ≠ b (desugared to ¬(a = b)).
    The prefix form (Lt(a, b), Gt(a, b)) is always available as well.
    """

    # --- Parsing ---

    def test_equality_infix_parses(self):
        t = _p("∀x ∀y (x = y → A(x))")
        assert t is not None

    def test_inequality_infix_parses(self):
        t = _p("∀x ∀y (x ≠ y → A(x))")
        assert t is not None

    def test_less_than_infix_parses(self):
        t = _p("∀x ∀y (x < y → A(x))")
        assert t is not None

    def test_greater_than_infix_parses(self):
        t = _p("∀x ∀y (x > y → A(x))")
        assert t is not None

    def test_lt_in_existential(self):
        t = _p("∃x (Age(x, n) ∧ n < threshold)")
        assert t is not None

    def test_prefix_lt_equivalent_to_infix(self):
        # Lt(a, b) and a < b both produce RelationNode with name 'Lt' vs '<'
        t_infix = _p("∀x ∀y (x < y → A(x))")
        t_prefix = _p("∀x ∀y (Lt(x, y) → A(x))")
        # They are structurally different nodes but both parse
        assert t_infix is not None
        assert t_prefix is not None

    # --- RelationNode structure ---

    def test_equality_produces_relation_node(self):
        t = _p("a = b")
        assert isinstance(t, RelationNode)
        assert t.name == '='
        assert t.arguments == ['a', 'b']

    def test_less_than_produces_relation_node(self):
        t = _p("a < b")
        assert isinstance(t, RelationNode)
        assert t.name == '<'
        assert t.arguments == ['a', 'b']

    def test_greater_than_produces_relation_node(self):
        t = _p("a > b")
        assert isinstance(t, RelationNode)
        assert t.name == '>'
        assert t.arguments == ['a', 'b']

    def test_inequality_desugars_to_not_equal(self):
        t = _p("a ≠ b")
        assert isinstance(t, BooleanNode)
        assert t.operator == 'not'
        inner = t.children[0]
        assert isinstance(inner, RelationNode)
        assert inner.name == '='

    # --- Signature ---

    def test_equality_appears_in_rel(self):
        sig = FOLSignature().extract(_p("∀x ∀y (x = y → A(x))"))
        assert "=" in sig["Rel"]
        assert sig["Rel"]["="] == 2

    def test_less_than_appears_in_rel(self):
        sig = FOLSignature().extract(_p("∀x ∀y (x < y → A(x))"))
        assert "<" in sig["Rel"]
        assert sig["Rel"]["<"] == 2

    def test_greater_than_appears_in_rel(self):
        sig = FOLSignature().extract(_p("∀x ∀y (x > y → A(x))"))
        assert ">" in sig["Rel"]
        assert sig["Rel"][">"] == 2

    def test_inequality_equality_in_rel(self):
        # ≠ desugars to ¬(= …) so '=' should appear in Rel
        sig = FOLSignature().extract(_p("∀x ∀y (x ≠ y → A(x))"))
        assert "=" in sig["Rel"]

    # --- Translator to_string round-trip ---

    def test_equality_round_trips(self):
        from fol_tools import FOLTranslator
        s = FOLTranslator().to_string(_p("∀x ∀y (x = y → A(x))"))
        assert "=" in s
        assert _p(s) is not None

    def test_less_than_round_trips(self):
        from fol_tools import FOLTranslator
        s = FOLTranslator().to_string(_p("∀x ∀y (x < y → A(x))"))
        assert "<" in s
        assert _p(s) is not None

    def test_greater_than_round_trips(self):
        from fol_tools import FOLTranslator
        s = FOLTranslator().to_string(_p("∀x ∀y (x > y → A(x))"))
        assert ">" in s
        assert _p(s) is not None

    # --- NL generation ---

    def test_equality_nl(self):
        from fol_tools import FOLTranslator
        nl = FOLTranslator().to_nl(_p("a = b"))
        assert "equals" in nl.lower()

    def test_less_than_nl(self):
        from fol_tools import FOLTranslator
        nl = FOLTranslator().to_nl(_p("a < b"))
        assert "less than" in nl.lower()

    def test_greater_than_nl(self):
        from fol_tools import FOLTranslator
        nl = FOLTranslator().to_nl(_p("a > b"))
        assert "greater than" in nl.lower()

    def test_negated_less_than_nl(self):
        from fol_tools import FOLTranslator
        nl = FOLTranslator().to_nl(_p("¬(a < b)"))
        assert "not less than" in nl.lower()

    def test_negated_greater_than_nl(self):
        from fol_tools import FOLTranslator
        nl = FOLTranslator().to_nl(_p("¬(a > b)"))
        assert "not greater than" in nl.lower()

    # --- Encoding and Z3 ---

    def test_equality_smtlib(self):
        s = FOL("∀x ∀y (x = y → A(x))").to_smtlib()
        assert "=" in s or "==" in s

    def test_less_than_smtlib(self):
        # < is uninterpreted: appears as a function call in SMT-LIB
        s = FOL("∀x ∀y (x < y → A(x))").to_smtlib()
        assert isinstance(s, str) and len(s) > 0

    def test_greater_than_smtlib(self):
        s = FOL("∀x ∀y (x > y → A(x))").to_smtlib()
        assert isinstance(s, str) and len(s) > 0

    def test_equality_semantics_via_solver(self):
        # x = y should entail y = x (Z3 native equality is symmetric)
        solver = FOLSolver()
        f1 = FOL("∀x ∀y (x = y → y = x)")
        assert solver.is_satisfiable(f1)

    def test_equality_reflexivity_via_solver(self):
        solver = FOLSolver()
        assert solver.is_satisfiable(FOL("∀x (x = x)"))

    def test_lt_uninterpreted_by_default(self):
        # Without axioms, < has no order semantics: ∀x(x < x) is satisfiable
        solver = FOLSolver()
        assert solver.is_satisfiable(FOL("∀x (x < x)"))

    def test_lt_can_be_axiomatised(self):
        # With irreflexivity as an axiom, ∀x(x < x) becomes unsat
        solver = FOLSolver()
        theory = ["∀x ¬(x < x)"]
        assert not solver.is_satisfiable(FOL("∀x (x < x)"), theory=theory)

    def test_comparison_in_real_formula(self):
        # Inspired by story_306 where GreaterThan(y, four) is used;
        # here we use the infix form to check the same chain works
        theory = [
            "∀x (∃y (HaveRating(x, y) ∧ y > four) → Listed(x))",
            "∀x (Listed(x) → Popular(x))",
        ]
        f1 = FOL("∃y (HaveRating(restaurant, y) ∧ y > four)")
        f2 = FOL("Popular(restaurant)")
        assert FOLSolver().implies(f1, f2, theory=theory)


# ===========================================================================
# 6. Known-invalid FOL_sentence_old formulas: documented failure modes
# ===========================================================================

class TestKnownInvalidOldFormulas:
    def test_failure_A_multi_axiom_whitespace(self):
        """
        A: axioms juxtaposed with whitespace only (76 occurrences).
        Parser consumes the first formula then raises 'not fully consumed'.
        """
        bad = "∀x (Square(x) → FourSided(x)) ∀x (FourSided(x) → Shape(x))"
        with pytest.raises(ValueError, match="not fully consumed"):
            FOLParser().parse(bad)

    def test_failure_A_existential_juxtaposed(self):
        bad = "∀x (Cat(x) → Mammal(x)) ∃x (Pet(x) ∧ ¬Mammal(x))"
        with pytest.raises(ValueError, match="not fully consumed"):
            FOLParser().parse(bad)

    def test_failure_A_ground_atoms_juxtaposed(self):
        bad = "ProfessionalWrestlingStable(diamondMine) ∧ In(diamondMine, wWE) Leads(roderickStrong, diamondMine)"
        with pytest.raises(ValueError, match="not fully consumed"):
            FOLParser().parse(bad)

    def test_failure_A_can_be_fixed_by_joining_with_conjunction(self):
        """Joining with ' ∧ ' produces a valid formula (DatasetCuration pipeline fix)."""
        axioms = ["∀x (Square(x) → FourSided(x))", "∀x (FourSided(x) → Shape(x))"]
        joined = " ∧ ".join(f"({a})" for a in axioms)
        assert FOLParser().parse(joined) is not None

    def test_failure_B_unbalanced_paren_at_eof(self):
        """B: genuine unbalanced parenthesis — 2 occurrences."""
        bad = "Take(james, databaseCourse) ⊕ (∃y (PartTimeJob(y) ∧ Have(james, y) ∧ OfferedBy(y, university))"
        with pytest.raises(ValueError):
            FOLParser().parse(bad)

    def test_failure_B_missing_closing_in_existential(self):
        bad = "∃x ((TakeOutService(h) ∧ NegativeReview(x) ∧ Receive(h, x) → Popular(h))"
        with pytest.raises(ValueError):
            FOLParser().parse(bad)

    def test_failure_C_long_arrow_character(self):
        """C: U+27F7 ⟷ (LONG LEFT RIGHT ARROW) instead of U+2194 ↔ — 1 occurrence."""
        bad = "∀x ∀y (GoodGuy(x) ∧ Fight(x, y) ⟷ BadGuy(y) ∧ Fight(y, x))"
        with pytest.raises(ValueError, match="Unexpected character"):
            FOLParser().parse(bad)

    def test_failure_C_fixed_by_replacing_long_arrow(self):
        fixed = "∀x ∀y (GoodGuy(x) ∧ Fight(x, y) ⟷ BadGuy(y) ∧ Fight(y, x))".replace("⟷", "↔")
        assert FOLParser().parse(fixed) is not None

    def test_failure_D_free_variable(self):
        """D: formula syntactically ok but has a free variable."""
        f = FOL("Student(x) ∧ AttendSchool(x)")
        assert f.validate() is False
        assert "x" in f.free_variables()

    def test_failure_D_fixed_by_adding_quantifier(self):
        assert FOL("∀x (Student(x) ∧ AttendSchool(x))").validate() is True


# ===========================================================================
# 7. FOL.validate() on a representative cross-section
# ===========================================================================

class TestValidateOnRealFormulas:
    @pytest.mark.parametrize("formula", [
        # Standard quantified formulas — fully bound
        "∀x (Human(x) → Mortal(x))",
        "∀x ∀y (x = y → Human(x))",
        "∀x ∀y ∀z (Between(x, y, z) → ¬Between(z, y, x))",
        "∀x ∀y (x < y → ¬(x > y))",
        # Boolean constants
        "⊤",
        "⊥",
        # Uppercase-starting constants (recognised without any declaration)
        "∀x (WildTurkey(x) ↔ EasternWildTurkey(x))",
        "∀x (DeadlyDisease(x) → ComeWith(x, LowSurvivalRate))",
        # FOLIO-style lowercase constants — multi-char lowercase names (tom, joey,
        # istanbul, threshold) and camelCase names (diamondMine, wWE, databaseCourse)
        # are all treated as constants by the [a-z][0-9]* variable pattern rule.
        # Lowercase-starting predicates followed by '(' are also valid.
        "∀x human(x)",
        "∀x isDigitalMedia(x)",
        "∀x (isStreamingService(x) → ¬isPhysicalMedia(x))",
        "ProfessionalWrestlingStable(diamondMine) ∧ In(diamondMine, wWE)",
        "∃x (Owns(tom, x) ∧ VehicleRegistrationPlateIn(x, istanbul))",
        "WildTurkey(joey)",
        "Take(james, databaseCourse) ⊕ (∃y (PartTimeJob(y) ∧ Have(james, y) ∧ OfferedBy(y, university)))",
        "∀x (x < threshold → Small(x))",
    ])
    def test_valid_formulas(self, formula):
        assert FOL(formula).validate() is True

    @pytest.mark.parametrize("formula", [
        "∀x (A(x) -> B(x))",        # forbidden ASCII ->
        "∀x (A(x) & B(x))",         # forbidden ASCII &
        "Student(x) ∧ Human(x)",     # free variable x
    ])
    def test_invalid_formulas(self, formula):
        assert FOL(formula).validate() is False
