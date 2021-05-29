#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from dataclasses import field
from typing import List, Optional, Union

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ValueOp:
    value: str


@dataclass(frozen=True)
class CallLikeOp:
    name: str


@dataclass(frozen=True)
class BuildStructOp:
    op_schema: str
    op_fields: List[str]
    empty_base: bool
    push_go: bool


# NOTE: Currently, since these three ops have different schema and they are not convertible with each other,
# it is type-safe to use `Union` for deserialization.
# See explanation https://pydantic-docs.helpmanual.io/usage/types/#unions.
# If there are two ops sharing the same schema or are convertible between each other, we need to chante
# `Op` to a dataclass and explicitly define a `op_type` field.
Op = Union[ValueOp, CallLikeOp, BuildStructOp]


@dataclass(frozen=True)
class Expression:
    id: str
    op: Op
    arg_ids: List[str] = field(default_factory=list)
    # TODO maybe parse into a more friendly format than Sexp str
    type_args: Optional[List[str]] = None  # type: ignore # Recursive type
    # TODO maybe parse into a more friendly format than Sexp str
    type: Optional[str] = None  # type: ignore # Recursive type


@dataclass(frozen=True)
class Program:
    expressions: List[Expression]
