"""
Arithmetic evaluation without eval().

Only integer literals and the binary operators + - * are accepted; anything
else (names, calls, attributes, powers, huge numbers) raises ValueError.
"""
from __future__ import annotations

import ast
import operator

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
_MAX_NODES = 16
_MAX_ABS = 10**6


def safe_eval(expr: str) -> int:
    expr = expr.replace("x", "*").replace("X", "*").replace("×", "*").strip()
    if not expr or len(expr) > 40:
        raise ValueError("expression empty or too long")
    tree = ast.parse(expr, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ValueError("expression too complex")

    def ev(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            value = _OPS[type(node.op)](ev(node.left), ev(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = -ev(node.operand)
        else:
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if abs(value) > _MAX_ABS:
            raise ValueError("result out of range")
        return value

    return ev(tree)
