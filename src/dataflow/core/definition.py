from dataclasses import dataclass
from typing import List

from dataflow.core.program import TypeName


@dataclass(frozen=True)
class Definition:
    """A function signature. For example,
    Definition("foo", ["T"], [TypeName("Long"), TypeName("T")], TypeName("Double"))
    would be
    T = TypeVar("T")
    def foo(arg1: Long, arg2: T) -> Double:

    in Python.

    This class is currently only used in type_inference.py, but we might use
    it elsewhere too."""

    name: str
    type_args: List[str]
    args: List[TypeName]
    type: TypeName
