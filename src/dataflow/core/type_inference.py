import json
from abc import ABC
from dataclasses import dataclass, replace
from typing import Dict, Optional, List, Tuple, Set

from dataflow.core.definition import Definition
from dataflow.core.program import Program, Op, TypeName, roots_and_reentrancies, \
    Expression, CallLikeOp, ValueOp, BuildStructOp


class Type(ABC):
    pass


class TypeVariable(Type):
    pass


class NamedTypeVariable(TypeVariable):
    name: str

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f'$_{self.name}'


class AnonTypeVariable(TypeVariable):
    pass

    def __repr__(self):
        return f'$_{hex(id(self))}'


@dataclass(frozen=True)
class TypeApplication(Type):
    constructor: str
    args: List[Type]

    def __repr__(self):
        if len(self.args) == 0:
            return f'{self.constructor}'
        else:
            return f'{self.constructor}[{",".join([arg.__repr__() for arg in self.args])}]'


@dataclass(frozen=False)
class Computation:
    op: Optional[str]
    id: str
    args: List["Computation"]
    type_args: List[NamedTypeVariable]
    type: Type


def infer_types(program: Program, library: Dict[str, Definition]) -> Program:
    id_to_expr = {expr.id: expr for expr in program.expressions}

    substitutions: Dict[TypeVariable, Type] = {}
    computation = _to_computation(program, library, substitutions, id_to_expr)

    inferred_computation = _infer_types_rec(computation, library, substitutions)
    closed_list: Set[str] = set()
    id_to_comp: Dict[str, Computation] = {}

    def set_types_rec(comp: Computation):
        if comp.id in closed_list:
            pass
        else:
            id_to_comp[comp.id] = comp
            closed_list.add(comp.id)
            comp.type = _apply_substitutions(comp.type, substitutions)
            for arg in comp.args:
                set_types_rec(arg)

    set_types_rec(inferred_computation)
    new_expressions = []
    for expr in program.expressions:
        if expr.id not in id_to_comp:
            print("here")
        comp = id_to_comp[expr.id]
        new_expressions.append(replace(expr, type=comp.type.args[-1], type_args=[_apply_substitutions(t, substitutions) for t in comp.type_args]))
    return Program(new_expressions)


def _infer_types_rec(computation: Computation, library: Dict[str, Definition],
                     substitutions: Dict[TypeVariable, Type]) -> Computation:

    inferred_args = [_infer_types_rec(arg, library, substitutions).type.args[-1] for arg in computation.args]
    if len(inferred_args) == 0:
        inferred_args = [TypeApplication("Unit", [])]

    actual_type = TypeApplication("Lambda", inferred_args + [computation.type.args[-1]])
    computation.type = _unify(actual_type, computation.type, substitutions)
    return computation


def _to_computation(program: Program, library: Dict[str, Definition], substitutions: Dict[TypeVariable, Type], id_to_expr: Dict[str, Expression]) -> Computation:
    closed_list: Dict[str, Computation] = {}


    def rec(expression: Expression) -> Computation:
        if expression.id in closed_list:
            return closed_list[expression.id]
        rec_args = [rec(id_to_expr[arg_expr_id]) for arg_expr_id in expression.arg_ids]
        if isinstance(expression.op, CallLikeOp):
            op = expression.op.name
        elif isinstance(expression.op, ValueOp):
            type_name = json.loads(expression.op.value)["schema"]
            op = type_name
        elif isinstance(expression.op, BuildStructOp):
            assert f"BuildStructOp {expression.op} not supported in type inference"
        defn = library[op]
        declared_type_args_list = [NamedTypeVariable(arg_name) for arg_name in
                              defn.type_args]
        declared_type_args = {var.name: var for var in declared_type_args_list}
        defn_type = _definition_to_type(defn, declared_type_args)
        ascribed_return_type = _type_name_to_type(expression.type, {}) if expression.type else AnonTypeVariable()
        assert expression.type_args is None or len(expression.type_args) == len(defn.type_args), f"Must either have no type arguments or the same number as the function declaration, but got {expression.type_args} and {defn.type_args}"
        if expression.type_args:
            for (ascribed_type_arg, type_var) in zip(expression.type_args, declared_type_args_list):
                substitutions[type_var] = _type_name_to_type(ascribed_type_arg, {})
        ascribed_arg_types = [AnonTypeVariable() for arg in defn.args] if len(defn.args) > 0 else [TypeApplication("Unit", [])]
        ascribed_type = TypeApplication("Lambda", ascribed_arg_types + [ascribed_return_type])
        comp_type = _unify(ascribed_type, defn_type, substitutions)
        return Computation(op, expression.id, rec_args, declared_type_args_list, comp_type)
    (roots, _) = roots_and_reentrancies(program)
    assert len(roots) == 1, f"Expected Program to have a single root, got {roots} in {program}"
    root_id = list(roots)[0]
    return rec(id_to_expr[root_id])


def _definition_to_type(definition: Definition,
                        declared_type_args: Dict[str, TypeVariable]) -> Type:
    return_type = _type_name_to_type(definition.type, declared_type_args)
    arg_types = [_type_name_to_type(arg, declared_type_args) for arg in definition.args]
    if len(arg_types) == 0:
        arg_types = [TypeApplication("Unit", [])]
    return TypeApplication("Lambda", arg_types + [return_type])


def _type_name_to_type(name: TypeName,
                       declared_type_args: Dict[str, TypeVariable]) -> Type:
    if name is None:
        return AnonTypeVariable()
    if len(name.type_args) == 0:
        if name.base in declared_type_args:
            return declared_type_args[name.base]
        else:
            return TypeApplication(name.base, [])
    else:
        args = [_type_name_to_type(arg, declared_type_args) for arg in name.type_args]
        return TypeApplication(name.base, args)


def _apply_substitutions(t: Type, substitutions: Dict[TypeVariable, Type]):
    if isinstance(t, TypeVariable):
        if t in substitutions:
            return _apply_substitutions(substitutions[t], substitutions)
        else:
            return t
    if isinstance(t, TypeApplication):
        return TypeApplication(t.constructor, [_apply_substitutions(arg, substitutions) for arg in t.args])


def _unify(t1: Type, t2: Type, substitutions: Dict[TypeVariable, Type]) -> Type:
    t1 = _apply_substitutions(t1, substitutions)
    t2 = _apply_substitutions(t2, substitutions)
    if t1 == t2:
        return t1
    # If either t1 or t2 is a type variable, and it does not occur in the other type,
    # then produce the substitution that binds it to the other type.
    elif isinstance(t2, TypeVariable) and not isinstance(t1, TypeVariable):
        return _unify(t2, t1, substitutions)
    elif isinstance(t1, TypeVariable) and not _occurs(t2, t1):
        substitutions[t1] = t2
        return t2
    # If we have two type applications of the same arity,
    # then unify the types and the corresponding type arguments.
    # The overall substitution that unifies the two applications is just the composition
    # of all the resulting substitutions together.
    elif isinstance(t1, TypeApplication) and isinstance(t2, TypeApplication) \
            and t1.constructor == t2.constructor and len(t1.args) == len(t2.args):
        unified_args = [
            _unify(arg1, arg2, substitutions) for (arg1, arg2) in zip(t1.args, t2.args)
        ]
        return TypeApplication(t1.constructor, unified_args)
    # All other cases result in a unification failure.
    else:
        raise Exception(f"Can't unify {t1} and {t2}")


def _occurs(t: Type, var: TypeVariable) -> bool:
    if t == var:
        return True
    elif isinstance(t, TypeApplication):
        return any(_occurs(a, var) for a in t.args)
    else:
        return False
