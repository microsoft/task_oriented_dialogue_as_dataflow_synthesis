from dataclasses import dataclass
from typing import Dict, List, Tuple

from dataflow.core.lispress import (
    META_CHAR,
    Lispress,
    lispress_to_type_name,
    parse_lispress,
)
from dataflow.core.program import TypeName


@dataclass(frozen=True)
class Definition:
    """A function signature. For example,
    Definition("foo", ["T"], [("arg1", TypeName("Long")), ("arg2", TypeName("T"))], TypeName("Double"))
    would be

    T = TypeVar("T")
    def foo(arg1: Long, arg2: T) -> Double:
        pass

    in Python, and

    (def ^(T) foo (^Long arg1 ^T arg2) ^Double ???)

    in Lispress. The ??? is the "body" of the def, much like `pass` in Python.
    It's slightly easier there's always a body because that's where return
    type annotations live right now.

    This class is currently only used in type_inference.py, but we might use
    it elsewhere too."""

    name: str
    type_params: List[str]
    params: List[Tuple[str, TypeName]]
    return_type: TypeName

    def __post_init__(self):
        assert len(set(name for name, typeName in self.params)) == len(
            self.params
        ), f"Duplicate arg names found for {self.name}. Args were {self.params}"


def lispress_library_to_library(lispress_str: str) -> Dict[str, Definition]:
    """Parses a list of lispress function defs into a indexed collection of Definitions.
    For example an input might look like

    (def + (^Long a ^Long b) ^Long ???)
    (package my.namespace
      (def - (^Long a ^Long b) ^Long ???)
    )

    The returned library would contain an entry for '+' and 'my.namespace.-'.
    """
    # The files are at flat list of global files and namespaces packages.
    # Wrap everything in parens to make it parse as a single expression.
    sexp = parse_lispress("(" + lispress_str + ")")
    assert isinstance(
        sexp, list
    ), f"Expected list of S-Expressions in file {lispress_str}"
    res: Dict[str, Definition] = {}
    for def_or_package in sexp:
        if isinstance(def_or_package, list) and def_or_package[0] == "def":
            # def in the global namespace
            defn = _def_to_definition(def_or_package, namespace="")
            res[defn.name] = defn
        elif isinstance(def_or_package, list) and def_or_package[0] == "package":
            assert isinstance(
                def_or_package, list
            ), f"Expected list of S-Expressions in package {def_or_package}"
            (unused_package_kw, package_name, *defs) = def_or_package
            for lispress_def in defs:
                defn = _def_to_definition(lispress_def, namespace=package_name)
                res[defn.name] = defn
    return res


def _def_to_definition(lispress_def: Lispress, namespace: str) -> Definition:
    (unused_keyword, func_name, param_list, body) = lispress_def
    if isinstance(func_name, list) and func_name[0] == META_CHAR:
        (unused_meta, type_params, actual_func_name) = func_name
    else:
        actual_func_name = func_name
        type_params = []

    assert (
        isinstance(body, list) and body[0] == META_CHAR
    ), f"Invalid function body {body}"
    (unused_meta, return_type, unused_body) = body

    params = [
        _parse_param(param)
        for param in param_list  # Skip typeclass constraints for now
        if not (isinstance(param, list) and param[0] == "using")
    ]
    namespace_prefix = namespace + "." if len(namespace) > 0 else ""
    return Definition(
        namespace_prefix + actual_func_name,
        type_params,
        params,
        lispress_to_type_name(return_type),
    )


def _parse_param(param: Lispress) -> Tuple[str, TypeName]:
    assert (
        isinstance(param, list) and param[0] == META_CHAR
    ), f"Invalid function param {param}"
    (unused_meta, type_ascription, param_name_maybe_with_default) = param

    if isinstance(param_name_maybe_with_default, str):
        param_name = param_name_maybe_with_default
    elif isinstance(param_name_maybe_with_default, list):
        (param_name, unused_default) = param_name_maybe_with_default
    # Option[T] is an optional argument, so just make it T
    if isinstance(type_ascription, list) and type_ascription[0] == "Option":
        type_ascription = type_ascription[1]
    return param_name, lispress_to_type_name(type_ascription)
