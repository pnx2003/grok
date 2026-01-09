# 复用原有二元运算数据生成逻辑（直接拷贝/导入原有代码）
import numpy as np
import itertools
from sympy.combinatorics import Permutation
from sympy import Mod

# 原有常量（若原有代码有定义，直接import，此处为兜底）
MODULUS = 10
NUMS = list(range(MODULUS))

class OperationDataGenerator:
    @classmethod
    def render(cls, elem):
        """统一渲染元素为字符串（适配Permutation/Mod/普通数字）"""
        if isinstance(elem, Permutation):
            return str(elem.array_form)
        elif isinstance(elem, Mod):
            return f"{elem.args[0]}_mod_{elem.args[1]}"
        else:
            return str(elem)

    @classmethod
    def _make_binary_operation_data(cls, operator: str, operands=None) -> list:
        """完全复用原有数据生成逻辑（直接拷贝原有函数）"""
        if operator == "s5":
            operands = operands or list(range(5))
            elems = map(np.array, itertools.permutations(operands))
            tuples = itertools.product(elems, repeat=2)
        elif operator in ["s5conj", "s5aba"]:
            operands = operands or list(range(5))
            elems = map(Permutation, itertools.permutations(operands))
            tuples = itertools.product(elems, repeat=2)
        elif "_mod_" in operator:
            modulo = int(operator.split("_mod_")[-1])
            elems = [Mod(i, modulo) for i in range(modulo)]
            tuples = itertools.product(elems, repeat=2)
        else:
            operands = operands or NUMS
            tuples = itertools.product(operands, repeat=2)

        eqs = []
        for a, b in tuples:
            if operator == "/":
                if b == 0:
                    continue
                else:
                    c = a
                    a = (b * c) % MODULUS
            elif operator == "s5":
                c = b[a]
            elif operator == "s5conj":
                c = a * b * (a.__invert__())
            elif operator == "s5aba":
                c = a * b * a
            elif operator == "+*":
                if a % 2 == 0:
                    c = (a + b) % MODULUS
                else:
                    c = (a * b) % MODULUS
            elif operator == "+-":
                if a % 2 == 0:
                    c = (a + b) % MODULUS
                else:
                    c = (a - b) % MODULUS
            elif "_mod_" in operator:
                expression = operator.split("_mod_")[0]
                function = eval(f"lambda x, y: ({expression})")
                c = function(a, b)
            else:
                c = eval(f"({a} {operator} {b}) % {MODULUS}")
            
            eqs.append({
                "a": cls.render(a),
                "op": operator,
                "b": cls.render(b),
                "c": cls.render(c)
            })
        return eqs