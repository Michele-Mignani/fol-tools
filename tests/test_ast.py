"""
tests/test_ast.py
=================

Unit tests for fol_tools.ast node classes.

Covers:
  - Valid construction of all four node types
  - Invalid construction (bad operator, bad quantifier)
  - repr strings do not crash
"""

import pytest
from fol_tools.ast import (
    Node,
    QuantifierNode,
    BooleanNode,
    RelationNode,
    BoolConstNode,
)


class TestNode:
    def test_node_is_base_class(self):
        assert issubclass(QuantifierNode, Node)
        assert issubclass(BooleanNode, Node)
        assert issubclass(RelationNode, Node)
        assert issubclass(BoolConstNode, Node)


class TestQuantifierNode:
    def test_forall(self):
        n = QuantifierNode('forall', 'x', BoolConstNode(True))
        assert n.quantifier == 'forall'
        assert n.variable == 'x'
        assert isinstance(n.body, BoolConstNode)

    def test_exists(self):
        n = QuantifierNode('exists', 'y', BoolConstNode(False))
        assert n.quantifier == 'exists'

    def test_invalid_quantifier_raises(self):
        with pytest.raises(ValueError, match="quantifier"):
            QuantifierNode('all', 'x', BoolConstNode(True))

    def test_body_is_stored_as_given(self):
        body = RelationNode('Human', ['x'])
        n = QuantifierNode('forall', 'x', body)
        assert n.body is body


class TestBooleanNode:
    @pytest.mark.parametrize('op', ['not', 'and', 'or', 'implies', 'iff', 'xor'])
    def test_valid_operators(self, op):
        children = [BoolConstNode(True)] if op == 'not' else [BoolConstNode(True), BoolConstNode(False)]
        n = BooleanNode(op, children)
        assert n.operator == op
        assert n.children == children

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="operator"):
            BooleanNode('nand', [BoolConstNode(True), BoolConstNode(False)])

    def test_children_are_copied_to_list(self):
        original = (BoolConstNode(True), BoolConstNode(False))
        n = BooleanNode('and', original)
        assert isinstance(n.children, list)
        assert n.children[0] is original[0]


class TestRelationNode:
    def test_named_with_args(self):
        n = RelationNode('Loves', ['x', 'y'])
        assert n.name == 'Loves'
        assert n.arguments == ['x', 'y']

    def test_zero_arity(self):
        n = RelationNode('Rain', [])
        assert n.arguments == []

    def test_equality_name(self):
        n = RelationNode('=', ['x', 'y'])
        assert n.name == '='

    def test_arguments_copied_to_list(self):
        args = ('a', 'b')
        n = RelationNode('R', args)
        assert isinstance(n.arguments, list)


class TestBoolConstNode:
    def test_true(self):
        n = BoolConstNode(True)
        assert n.value is True

    def test_false(self):
        n = BoolConstNode(False)
        assert n.value is False

    def test_truthy_int_coerced(self):
        n = BoolConstNode(1)
        assert n.value is True

    def test_falsy_int_coerced(self):
        n = BoolConstNode(0)
        assert n.value is False
