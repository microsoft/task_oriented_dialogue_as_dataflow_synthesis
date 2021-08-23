import json
from abc import ABC
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Set, cast

from dataflow.core.definition import Definition
from dataflow.core.program import (
    BuildStructOp,
    CallLikeOp,
    Expression,
    Program,
    TypeName,
    ValueOp,
    roots_and_reentrancies,
)


@dataclass(frozen=True)
class TypeInferenceError(Exception):
    msg: str


class Type(ABC):
    pass


class TypeVariable(Type):
    pass


class NamedTypeVariable(TypeVariable):
    """A named type variable like T. Note that named type variables have scope:
    several different functions might use the name T but they are distinct type
    variables. As such, reference identity for NamedTypeVariables matters."""

    name: str

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"$_{self.name}"


class AnonTypeVariable(TypeVariable):
    """An anonymous type variable used to represent types that need to be inferred.
    Compared by reference equality."""

    pass

    def __repr__(self):
        return f"$_{hex(id(self))}"


@dataclass(frozen=True)
class TypeApplication(Type):
    """A type constructor with some arguments."""

    constructor: str
    args: Sequence[Type] = field(default_factory=list)

    def __repr__(self):
        if len(self.args) == 0:
            return f"{self.constructor}"
        else:
            return (
                f'{self.constructor}[{",".join([arg.__repr__() for arg in self.args])}]'
            )


@dataclass(frozen=False)
class Computation:
    """A representation of a single function application during type inference.
    Value literals are treated as functions from Unit to their final value."""

    # The name of the function or value literal.
    op: Optional[str]
    # For mapping back to Expression.id
    id: str
    # The arguments to the function
    args: List["Computation"]
    # Declared type arguments based on the signature of op (defined in a Definition).
    type_args: List[NamedTypeVariable]
    # The type of this function, always represented as a
    # TypeApplication("Lambda", arg_types + [return_type])
    # Mutated during type inference.
    type: TypeApplication

    @property
    def return_type(self) -> Type:
        """The output type of this function. Mutated during type inference"""
        return self.type.args[-1]


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
        comp = id_to_comp[expr.id]
        new_expressions.append(
            replace(
                expr,
                type=_type_to_type_name(comp.return_type),
                type_args=[
                    _type_to_type_name(_apply_substitutions(t, substitutions))
                    for t in comp.type_args
                ]
                if len(comp.type_args) > 0
                else None,
            )
        )
    return Program(new_expressions)


def _infer_types_rec(
    computation: Computation,
    library: Dict[str, Definition],
    substitutions: Dict[TypeVariable, Type],
) -> Computation:

    inferred_args = [
        _infer_types_rec(arg, library, substitutions).return_type
        for arg in computation.args
    ]

    actual_type = TypeApplication("Lambda", inferred_args + [computation.return_type])
    unified = _unify(actual_type, computation.type, substitutions)
    assert isinstance(
        unified, TypeApplication
    ), "Unification of lambdas should always produce a TypeApplication"
    computation.type = unified
    return computation


def _to_computation(
    program: Program,
    library: Dict[str, Definition],
    substitutions: Dict[TypeVariable, Type],
    id_to_expr: Dict[str, Expression],
) -> Computation:
    closed_list: Dict[str, Computation] = {}

    def rec(expression: Expression) -> Computation:
        if expression.id in closed_list:
            return closed_list[expression.id]
        rec_args = [rec(id_to_expr[arg_expr_id]) for arg_expr_id in expression.arg_ids]
        if isinstance(expression.op, CallLikeOp):
            op = expression.op.name
            defn = library[op]
            declared_type_args_list = [
                NamedTypeVariable(arg_name) for arg_name in defn.type_args
            ]
            declared_type_args = {var.name: var for var in declared_type_args_list}
            defn_type = _definition_to_type(defn, declared_type_args)
            assert expression.type_args is None or len(expression.type_args) == len(
                defn.type_args
            ), f"Must either have no type arguments or the same number as the function declaration, but got {expression.type_args} and {defn.type_args}"
        elif isinstance(expression.op, ValueOp):
            value_info = json.loads(expression.op.value)
            type_name = value_info["schema"]
            op = value_info["underlying"]
            defn = None
            declared_type_args_list = []

            def mk_primitive_constructor(p: str) -> TypeApplication:
                return TypeApplication("Lambda", [TypeApplication(p)])

            if type_name in ("String", "Long", "Number", "Boolean"):
                defn_type = mk_primitive_constructor(type_name)
            else:
                raise TypeInferenceError(f"Unknown primitive type {type_name}")

        elif isinstance(expression.op, BuildStructOp):
            assert f"BuildStructOp {expression.op} not supported in type inference"
        else:
            assert False, f"Unexpected op {expression.op}"

        ascribed_return_type = (
            _type_name_to_type(expression.type, {})
            if expression.type
            else AnonTypeVariable()
        )

        if expression.type_args:
            for (ascribed_type_arg, type_var) in zip(
                expression.type_args, declared_type_args_list
            ):
                substitutions[type_var] = _type_name_to_type(ascribed_type_arg, {})
        ascribed_arg_types = (
            [cast(Type, AnonTypeVariable()) for arg in defn.args] if defn else []
        )
        ascribed_type = TypeApplication(
            "Lambda", ascribed_arg_types + [ascribed_return_type]
        )
        comp_type = _unify(ascribed_type, defn_type, substitutions)
        assert isinstance(
            comp_type, TypeApplication
        ), "unification of Lambdas should always produce a TypeApplication"
        return Computation(
            op, expression.id, rec_args, declared_type_args_list, comp_type
        )

    (roots, _) = roots_and_reentrancies(program)
    assert (
        len(roots) == 1
    ), f"Expected Program to have a single root, got {roots} in {program}"
    root_id = list(roots)[0]
    return rec(id_to_expr[root_id])


def _definition_to_type(
    definition: Definition, declared_type_args: Dict[str, NamedTypeVariable]
) -> Type:
    return_type = _type_name_to_type(definition.type, declared_type_args)
    arg_types = [_type_name_to_type(arg, declared_type_args) for arg in definition.args]

    return TypeApplication("Lambda", arg_types + [return_type])


def _type_name_to_type(
    name: TypeName, declared_type_args: Dict[str, NamedTypeVariable]
) -> Type:
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


def _type_to_type_name(t: Type) -> TypeName:
    assert isinstance(
        t, TypeApplication
    ), f"_type_to_type_name expects a Type that is fully instantiated with no variables, but got {t}"
    return TypeName(t.constructor, [_type_to_type_name(a) for a in t.args])


def _apply_substitutions(t: Type, substitutions: Dict[TypeVariable, Type]):
    if isinstance(t, TypeVariable):
        if t in substitutions:
            return _apply_substitutions(substitutions[t], substitutions)
        else:
            return t
    elif isinstance(t, TypeApplication):
        return TypeApplication(
            t.constructor, [_apply_substitutions(arg, substitutions) for arg in t.args]
        )

    else:
        raise TypeInferenceError(f"Unknown type {t}")


def _unify(t1: Type, t2: Type, substitutions: Dict[TypeVariable, Type]) -> Type:
    t1 = _apply_substitutions(t1, substitutions)
    t2 = _apply_substitutions(t2, substitutions)
    if t1 == t2:
        return t1
    # If either t1 or t2 is a type variable, and it does not occur in the other type,
    # then produce the substitution that binds it to the other type.
    elif isinstance(t2, TypeVariable) and not isinstance(t1, TypeVariable):
        # Lol, pylint thinks that because t1 and t2 match the param names but are
        # out of order, there might be a bug.
        # pylint: disable=arguments-out-of-order
        return _unify(t2, t1, substitutions)
    elif isinstance(t1, TypeVariable) and not _occurs(t2, t1):
        substitutions[t1] = t2
        return t2

    # If we have two type applications of the same arity,
    # then unify the types and the corresponding type arguments.
    # The overall substitution that unifies the two applications is just the composition
    # of all the resulting substitutions together.
    elif (
        isinstance(t1, TypeApplication)
        and isinstance(t2, TypeApplication)
        and t1.constructor == t2.constructor
        and len(t1.args) == len(t2.args)
    ):
        unified_args = [
            _unify(arg1, arg2, substitutions) for (arg1, arg2) in zip(t1.args, t2.args)
        ]
        return TypeApplication(t1.constructor, unified_args)
    # All other cases result in a unification failure.
    else:
        raise TypeInferenceError(f"Can't unify {t1} and {t2}")


def _occurs(t: Type, var: TypeVariable) -> bool:
    if t == var:
        return True
    elif isinstance(t, TypeApplication):
        return any(_occurs(a, var) for a in t.args)
    else:
        return False
