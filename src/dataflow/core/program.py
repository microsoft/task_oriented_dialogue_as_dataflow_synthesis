#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass(frozen=True)
class ValueOp:
    value: str


@dataclass(frozen=True)
class CallLikeOp:
    name: str


@dataclass(frozen=True)
class BuildStructOp:
    op_schema: str
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
    type_args: List["TypeName"]


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
