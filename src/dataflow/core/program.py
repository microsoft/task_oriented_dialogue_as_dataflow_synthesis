#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from cached_property import cached_property


@dataclass(frozen=True)
class ValueOp:
    value: str


@dataclass(frozen=True)
class CallLikeOp:
    name: str


@dataclass(frozen=True)
class BuildStructOp:
    op_schema: str
    # Named arguments. None for positional arguments
    op_fields: List[Optional[str]]
    empty_base: bool
    push_go: bool


# NOTE: Currently, since these three ops have different schema and they are not convertible with each other,
# it is type-safe to use `Union` for deserialization.
# See explanation https://pydantic-docs.helpmanual.io/usage/types/#unions.
# If there are two ops sharing the same schema or are convertible between each other, we need to chante
# `Op` to a dataclass and explicitly define a `op_type` field.
Op = Union[ValueOp, CallLikeOp, BuildStructOp]


@dataclass(frozen=True)
class TypeName:
    base: str
    # Tuples preferred so TypeNames can be hashable
    type_args: Tuple["TypeName", ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        if len(self.type_args) == 0:
            return self.base
        else:
            return f'({self.base} {" ".join(a.__repr__() for a in self.type_args)})'


@dataclass(frozen=True)
class Expression:
    id: str
    op: Op
    type_args: Optional[List[TypeName]] = None
    type: Optional[TypeName] = None
    arg_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Program:
    expressions: List[Expression]

    @cached_property
    def expressions_by_id(self) -> Dict[str, Expression]:
        return {expression.id: expression for expression in self.expressions}


def roots_and_reentrancies(program: Program) -> Tuple[List[str], Set[str]]:
    """
    Returns ids of roots (expressions that never appear as arguments) and
    reentrancies (expressions that appear more than once as arguments).
    Now that `do` expressions get their own nodes, there should be exactly
    one root.
    """
    arg_counts = Counter(a for e in program.expressions for a in e.arg_ids)
    # ids that are never used as args
    roots = [e.id for e in program.expressions if e.id not in arg_counts]
    # args that are used multiple times as args
    reentrancies = {i for i, c in arg_counts.items() if c >= 2}
    return roots, reentrancies
