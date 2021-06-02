#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import re
from enum import Enum
from json import dumps
from typing import Any, List, Optional, Tuple

from dataflow.core.program import (
    BuildStructOp,
    CallLikeOp,
    Expression,
    TypeName,
    ValueOp,
)
from dataflow.core.sexp import Sexp

# revise args
ROOT_LOCATION = "rootLocation"
OLD_LOCATION = "oldLocation"
NEW = "new"

# BuildStructOp special arg
NON_EMPTY_BASE = "nonEmptyBase"

Idx = int


class OpType(Enum):
    """The type of an op."""

    Call = "Call"
    Struct = "Struct"
    Value = "#"


class DataflowFn(Enum):
    """Special Dataflow functions"""

    Find = "find"  # search
    Abandon = "abandon"
    Revise = "ReviseConstraint"
    Refer = "refer"
    RoleConstraint = "roleConstraint"
    Get = "get"  # access a member field of an object


def idx_str(idx: Idx) -> str:
    return f"[{idx}]"


def is_idx_str(s: str) -> bool:
    return s.startswith("[") and s.endswith("]")


def unwrap_idx_str(s: str) -> int:
    return int(s[1:-1])


def is_struct_op_schema(name: str) -> bool:
    """BuildStructOp schemas begin with a capital letter."""
    if len(name) == 0:
        return False
    return re.match(r"[A-Z]", name[0]) is not None


def get_named_args(e: Expression) -> List[Tuple[str, Optional[str]]]:
    """
    Gets a list of (arg_name, arg_id) pairs.
    If `e` is a BuildStructOp, then `arg_names` are its `fields`, otherwise
    they are the 0-indexed argument position.
    """
    if isinstance(e.op, BuildStructOp):
        bso = e.op
        # a non-empty BuildStructOp has an implicit 0-th field name
        zeroth_field = [] if bso.empty_base else [NON_EMPTY_BASE]
        fields = zeroth_field + list(bso.op_fields)
    else:
        fields = [f"arg{i}" for i in range(len(e.arg_ids))]
    return list(zip(fields, e.arg_ids))


def mk_constraint(
    tpe: str, args: List[Tuple[Optional[str], int]], idx: Idx,
) -> Tuple[Expression, Idx]:
    return mk_struct_op(schema=f"Constraint[{tpe.capitalize()}]", args=args, idx=idx)


def mk_equality_constraint(val: int, idx: Idx) -> Tuple[Expression, Idx]:
    return mk_call_op(name="?=", args=[val], idx=idx)


def mk_unset_constraint(idx: Idx) -> Tuple[Expression, Idx]:
    return mk_struct_op(schema="EmptyConstraint", args=[], idx=idx)


def mk_salience(tpe: str, idx: Idx) -> Tuple[List[Expression], Idx]:
    constraint_expr, constraint_idx = mk_constraint(tpe=tpe, args=[], idx=idx)
    salience_expr, idx = mk_call_op(
        name=DataflowFn.Refer.value, args=[constraint_idx], idx=constraint_idx
    )
    return [constraint_expr, salience_expr], idx


def mk_salient_action(idx: Idx) -> Tuple[List[Expression], Idx]:
    """ (roleConstraint #(Path "output")) """
    path_expr, path_idx = mk_value_op(schema="Path", value="output", idx=idx,)
    intension_expr, intension_idx = mk_call_op(
        name=DataflowFn.RoleConstraint.value, args=[path_idx], idx=path_idx,
    )
    return [path_expr, intension_expr], intension_idx


def mk_revise(
    root_location_idx: Idx, old_location_idx: Idx, new_idx: Idx, idx: Idx,
) -> Tuple[Expression, Idx]:
    """
    Revises the salient constraint satisfying the constraint at `old_location_idx`,
    in the salient computation satisfying the constraint at `root_location_idx`,
    with the constraint at `new_idx`.
    In Lispress:
    ```
    (Revise
      :rootLocation {root_location}
      :oldLocation {old_location}
      :new {new})
    """
    return mk_struct_op(
        schema=DataflowFn.Revise.value,
        args=[
            (ROOT_LOCATION, root_location_idx),
            (OLD_LOCATION, old_location_idx),
            (NEW, new_idx),
        ],
        idx=idx,
    )


def mk_revise_the_main_constraint(
    tpe: str, new_idx: Idx
) -> Tuple[List[Expression], Idx]:
    """
    Revises the salient constraint (on values of type `tpe`) in the salient action, with the
    constraint at `new_idx`.
    (An "action" is an argument of `Yield`).
    In Lispress:
    ```
    (ReviseConstraint
      :rootLocation (RoleConstraint :role #(Path "output"))
      :oldLocation (Constraint[Constraint[{tpe}]])
      :new {new})
    ```
    """
    salient_action_exprs, salient_action_idx = mk_salient_action(new_idx)
    old_loc_expr, old_loc_idx = mk_struct_op(
        schema=f"Constraint[Constraint[{tpe.capitalize()}]]",
        args=[],
        idx=salient_action_idx,
    )
    revise_expr, revise_idx = mk_revise(
        root_location_idx=salient_action_idx,
        old_location_idx=old_loc_idx,
        new_idx=new_idx,
        idx=old_loc_idx,
    )
    return salient_action_exprs + [old_loc_expr, revise_expr], revise_idx


def mk_struct_op(
    schema: str, args: List[Tuple[Optional[str], Idx]], idx: Idx,
) -> Tuple[Expression, Idx]:
    new_idx = idx + 1
    # args = dict(args)  # defensive copy
    base = next((v for k, v in args if k == NON_EMPTY_BASE), None)
    is_empty_base = base is None
    arg_names = [k for k, v in args]
    # nonEmptyBase always comes first
    arg_vals = ([] if is_empty_base else [base]) + [v for k, v in args]
    flat_exp = Expression(
        id=idx_str(new_idx),
        op=BuildStructOp(
            op_schema=schema,
            op_fields=arg_names,
            empty_base=is_empty_base,
            push_go=True,
        ),
        arg_ids=[idx_str(v) for v in arg_vals],
    )
    return flat_exp, new_idx


def mk_call_op(name: str, args: List[Idx], idx: Idx = 0) -> Tuple[Expression, Idx]:
    new_idx = idx + 1
    flat_exp = Expression(
        id=idx_str(new_idx),
        op=CallLikeOp(name=name),
        arg_ids=[idx_str(v) for v in args],
    )
    return flat_exp, new_idx


def mk_type_name(sexp: Sexp) -> TypeName:
    if isinstance(sexp, str):
        return TypeName(sexp, [])
    hd, *tl = sexp
    return TypeName(hd, [mk_type_name(e) for e in tl])


def mk_value_op(value: Any, schema: str, idx: Idx) -> Tuple[Expression, Idx]:
    my_idx = idx + 1
    dumped = dumps({"schema": schema, "underlying": value})
    expr = Expression(id=idx_str(my_idx), op=ValueOp(value=dumped))
    return expr, my_idx
