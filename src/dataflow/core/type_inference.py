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

    def __repr__(self):
        return f'{self.name}[{",".join([arg.__repr__() for arg in self.args])}]'


class AnonTypeVariable(TypeVariable):
    pass

    def __repr__(self):
        return f'$_{hex(id(self))}'


@dataclass(frozen=True)
class TypeApplication(Type):
    constructor: str
    args: List[Type]

    def __repr__(self):
        return f'{self.constructor}[{",".join([arg.__repr__() for arg in self.args])}]'


@dataclass(frozen=False)
class Computation:
    op: Optional[str]
    id: str
    type_args: List[Type]
    args: List["Computation"]
    type: Type


def infer_types(program: Program, library: Dict[str, Definition]) -> Program:
    id_to_expr = {expr.id: expr for expr in program.expressions}

    computation = _to_computation(program, id_to_expr)
    substitutions: Dict[TypeVariable, Type] = {}
    inferred_computation = _infer_types_rec(computation, library, AnonTypeVariable(), {}, substitutions)
    closed_list: Set[str] = set()
    id_to_comp: Dict[str, Computation] = {}

    def set_types_rec(comp: Computation):
        if comp.id in closed_list:
            pass
        else:
            id_to_comp[comp.id] = comp
            closed_list.add[comp.id]
            comp.type = _apply_substitutions(comp.type, substitutions)

    set_types_rec(inferred_computation)
    new_expressions = []
    for expr in program.expressions:
        new_expressions.append(replace(expr, type=computation.type, type_args=computation.type_args))
    return Program(new_expressions)


def _infer_types_rec(computation: Computation, library: Dict[str, Definition],
                     outer_type: Type, env: Dict[str, TypeVariable],
                     substitutions: Dict[TypeVariable, Type]) -> Computation:
    unified_outer_type = _unify(outer_type, computation.type, substitutions)
    defn = library[computation.op]
    declared_type_args = [(arg_name, NamedTypeVariable(arg_name)) for arg_name in
                          defn.type_args]
    if len(defn.type_args) == 0:
        local_env = env
    else:
        local_env = env.copy()  # TODO this could be sped up if we use a persistent map
        for declared_type_arg in defn.type_args:
            type_var = NamedTypeVariable(declared_type_arg)
            local_env[type_var] = type_var
    defn_type = _definition_to_type(defn, declared_type_args)
    inferred_args = [_infer_types_rec(arg, library, arg.type, local_env, substitutions).type for arg in computation.args]
    if len(inferred_args) == 0:
        inferred_args = [TypeApplication("Unit", [])]

    actual_type = TypeApplication("Lambda", inferred_args + [unified_outer_type])
    unified = _unify(actual_type, defn_type, substitutions)
    if computation.type:
        computation.type = _unify(computation.type, unified.args.last)
    else:
        computation.type = unified.args[-1]
    return computation


def _to_computation(program: Program, id_to_expr: Dict[str, Expression]) -> Computation:
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
        return Computation(op, expression.id, [_type_name_to_type(type_arg) for type_arg in expression.type_args], rec_args, _type_name_to_type(expression.type))
    (roots, _) = roots_and_reentrancies(program)
    assert len(roots) == 1, f"Expected Program to have a single root, got {roots} in {program}"
    root_id = list(roots)[0]
    return rec(id_to_expr[root_id])


def _definition_to_type(definition: Definition,
                        declared_type_args: Dict[str, TypeVariable]) -> Type:
    return_type = _type_name_to_type(definition.type, declared_type_args)
    arg_types = [_type_name_to_type(arg, declared_type_args) for arg in definition.args]
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
        return TypeApplication(t.constructor, [_apply_substitutions(arg) for arg in t.args])


def _unify(t1: Type, t2: Type, substitutions: Dict[TypeVariable, Type]) -> Type:
    # If either t1 or t2 is a type variable, and it does not occur in the other type,
    # then produce the substitution that binds it to the other type.
    if isinstance(t2, TypeVariable):
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
        return any(_occurs(a) for a in t.args)
    else:
        return False
