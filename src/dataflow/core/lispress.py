#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import json
import re
from collections import Counter
from dataclasses import replace
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Set, Tuple

from more_itertools import chunked

from dataflow.core.program import (
    BuildStructOp,
    CallLikeOp,
    Expression,
    Op,
    Program,
    TypeName,
    ValueOp,
)
from dataflow.core.program_utils import DataflowFn, Idx, OpType, get_named_args
from dataflow.core.program_utils import is_idx_str as is_express_idx_str
from dataflow.core.program_utils import (
    is_struct_op_schema,
    mk_call_op,
    mk_struct_op,
    mk_type_name,
    mk_value_op,
    unwrap_idx_str,
)
from dataflow.core.sexp import LEFT_PAREN, RIGHT_PAREN, Sexp, parse_sexp, sexp_to_str

# node label used for external references
EXTERNAL_LABEL = "ExternalReference"
# for `render_pretty`
NUM_INDENTATION_SPACES = 2
# keyword for introducing variable bindings
LET = "let"
# keyword for sequencing programs that have multiple statements
SEQUENCE = "do"
SEQUENCE_SEXP: List[Sexp] = [SEQUENCE]  # to help out mypy
# variables will be named `x0`, `x1`, etc., in the order they are introduced.
VAR_PREFIX = "x"
# named args are given like `(fn :name1 arg1 :name2 arg2 ...)`
NAMED_ARG_PREFIX = ":"
# values are rendered as `#(MySchema "json_dump_of_my_value")`
VALUE_CHAR = "#"
META_CHAR = "^"

# Lispress has lisp syntax, and we represent it as an s-expression
Lispress = Sexp


def try_round_trip(lispress_str: str) -> str:
    """
    If `lispress_str` is valid lispress, round-trips it to and from `Program`.
    This puts named arguments in alphabetical order and normalizes numbers
    so they all have exactly one decimal place.
    If it is not valid, returns the original string unmodified.
    """
    try:
        return _try_round_trip(lispress_str)
    except Exception:  # pylint: disable=W0703
        return lispress_str


def _try_round_trip(lispress_str: str) -> str:
    # round-trip to canonicalize
    lispress = parse_lispress(lispress_str)
    program, _ = lispress_to_program(lispress, 0)
    round_tripped = program_to_lispress(program)

    def normalize_numbers(exp: Lispress) -> "Lispress":
        if isinstance(exp, str):
            try:
                num = float(exp)
                return f"{num:.1f}"
            except ValueError:
                return exp
        else:
            return [normalize_numbers(e) for e in exp]

    return render_compact(strip_copy_strings(normalize_numbers(round_tripped)))


def strip_copy_strings(exp: Lispress) -> "Lispress":
    """
    Strips any whitespace in quoted strings (e.g. " 1   " -> "1")
    """
    if isinstance(exp, str):
        if len(exp) > 2 and exp[0] == '"' and exp[-1] == '"':
            return '"' + exp[1:-1].strip() + '"'
        else:
            return exp
    else:
        return [strip_copy_strings(e) for e in exp]


def program_to_lispress(program: Program) -> Lispress:
    """ Converts a Program to Lispress. """
    unsugared = _program_to_unsugared_lispress(program)
    sugared_gets = _sugar_gets(unsugared)
    return sugared_gets


def lispress_to_program(lispress: Lispress, idx: Idx) -> Tuple[Program, Idx]:
    """
    Converts Lispress to a Program with ids starting at `idx`.
    Returns the last id used along with the Program.
    """
    desugared_gets = _desugar_gets(lispress)
    return _unsugared_lispress_to_program(desugared_gets, idx)


def render_pretty(lispress: Lispress, max_width: int = 60) -> str:
    """
    Renders the expression on one or more lines, with adaptive linebreaks and
    standard lisp indentation.
    Attempts to keep lines below `max_width` when possible.
    Right inverse of `parse_lispress` (I.e. `parse_lispress(render_pretty(p)) == p`).
    E.g.:
    >>> lispress = ['describe', [':start', ['findNextEvent', ['Constraint[Event]', ':attendees', ['attendeeListHasRecipientConstraint', ['recipientWithNameLike', ['Constraint[Recipient]'], '#', ['PersonName', '"Elaine"']]]]]]]
    >>> print(render_pretty(lispress))
    (describe
      (:start
        (findNextEvent
          (Constraint[Event]
            :attendees (attendeeListHasRecipientConstraint
              (recipientWithNameLike
                (Constraint[Recipient])
                #(PersonName "Elaine")))))))
    """
    result = "\n".join(_render_lines(sexp=lispress, max_width=max_width))
    return result


def render_compact(lispress: Lispress) -> str:
    """
    Renders Lispress on a single line. Right inverse of `parse_lispress`.
    E.g.:
    >>> lispress = ['describe', [':start', ['findNextEvent', ['Constraint[Event]', ':attendees', ['attendeeListHasRecipientConstraint', ['recipientWithNameLike', ['Constraint[Recipient]'], '#', ['PersonName', '"Elaine"']]]]]]]
    >>> print(render_compact(lispress))
    (describe (:start (findNextEvent (Constraint[Event] :attendees (attendeeListHasRecipientConstraint (recipientWithNameLike (Constraint[Recipient]) #(PersonName "Elaine")))))))
    """
    return sexp_to_str(lispress)


def parse_lispress(s: str) -> Lispress:
    """
    Parses a Lispress string into a Lispress object.
    Inverse of `render_pretty` or `render_compact`.
    E.g.:
    >>> s = \
    "(describe" \
    "  (:start" \
    "    (findNextEvent" \
    "      (Constraint[Event]" \
    "        :attendees (attendeeListHasRecipientConstraint" \
    "          (recipientWithNameLike" \
    "            (Constraint[Recipient])" \
    '            #(PersonName "Elaine")))))))'
    >>> parse_lispress(s)
    ['describe', [':start', ['findNextEvent', ['Constraint[Event]', ':attendees', ['attendeeListHasRecipientConstraint', ['recipientWithNameLike', ['Constraint[Recipient]'], '#', ['PersonName', '"Elaine"']]]]]]]
    """
    return parse_sexp(s)


def _group_named_args(lines: List[str]) -> List[str]:
    """
    Helper function for `_render_lines`.
    Joins `:name` and `argument` lines into a single line.
    """
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_named_arg(line):
            result.append(" ".join(lines[i : i + 2]))
            i += 2
        else:
            result.append(line)
            i += 1
    return result


def _render_lines(sexp: Lispress, max_width: int) -> List[str]:
    """Helper function for `render_pretty`."""
    compact = render_compact(sexp)
    if isinstance(sexp, str) or len(sexp) <= 1 or len(compact) <= max_width:
        return [compact]
    else:
        fn, *args = sexp
        if fn == VALUE_CHAR:
            assert len(args) == 1, "# Value expressions must have one argument"
            lines = _render_lines(args[0], max_width=max_width)
            lines[0] = VALUE_CHAR + lines[0]
            return lines
        elif fn == META_CHAR:
            assert len(args) == 2, "^ Meta expressions must have one argument"
            lines = _render_lines(args[0], max_width=max_width)
            lines.extend(_render_lines(args[1], max_width=max_width))
            lines[0] = META_CHAR + lines[0]
            return lines
        else:
            prefix = " " * NUM_INDENTATION_SPACES
            fn_lines = _render_lines(fn, max_width=max_width)
            arg_lines = _group_named_args(
                [
                    line
                    for arg in args
                    for line in _render_lines(arg, max_width=max_width)
                ]
            )
            lines = fn_lines + [prefix + line for line in arg_lines]
            lines[0] = LEFT_PAREN + lines[0]
            lines[-1] = lines[-1] + RIGHT_PAREN
            return lines


def _idx_to_var_str(idx: int) -> str:
    return f"{VAR_PREFIX}{idx}"


def _is_idx_str(s: str) -> bool:
    return len(s) > 1 and s.startswith(VAR_PREFIX) and s[len(VAR_PREFIX) :].isnumeric()


def _is_named_arg(name: str) -> bool:
    return name.startswith(NAMED_ARG_PREFIX)


def _named_arg_to_key(name: str) -> str:
    """Strips an initial `KEYWORD_PREFIX` off of `name`. Left-inverse of `key_to_named_arg`."""
    assert _is_named_arg(
        name
    ), f"named arg must start with '{NAMED_ARG_PREFIX}': {name}"
    return name[len(NAMED_ARG_PREFIX) :]


def _key_to_named_arg(key: str) -> str:
    """Adds `KEYWORD_PREFIX` to the start of `key`. Right-inverse of `named_arg_to_key`."""
    return f"{NAMED_ARG_PREFIX}{key}"


def op_to_lispress(op: Op) -> Lispress:
    """
    Converts `op` to lispress in a way that can be undone (see `seq_to_program`).
    Note that for struct and call ops, op just has the name of the op (doesn't include args).
    """
    if isinstance(op, BuildStructOp):
        op_schema = op.op_schema
        assert is_struct_op_schema(
            op_schema
        ), f"BuildStructOp schemas must begin with a capital letter ({op_schema})."
        return op_schema
    elif isinstance(op, CallLikeOp):
        op_name = op.name
        assert not is_struct_op_schema(
            op_name
        ), "CallLikeOp ops must not begin with a capital letter."
        return op_name
    elif isinstance(op, ValueOp):
        value = json.loads(op.value)
        schema = value.get("schema")
        underlying = value.get("underlying")
        # Long literals look like 2L
        if schema == "Long":
            return str(underlying) + "L"
        else:
            assert isinstance(
                underlying, (bool, str, int, float, list)
            ), f"Can't handle JSON dicts as value literals anymore, got {underlying}"
            assert not (
                isinstance(underlying, list) and len(underlying) > 0
            ), f"Can only handle empty list JSON value literals for backwards compatibility, but got {underlying}"
            underlying_json_str = json.dumps(underlying)
            if schema in ("Number", "String", "Boolean"):
                # Numbers and strings were typed in Calflow 1.0 (e.g. #(Number 1),
                # #(String "foo"), in Calflow 2.0, any bare number parseable as a float
                # or int is interpreted as Number, any quoted string is interpreted
                # as a string, and any boolean is interpreted as a boolean. This means
                # that we drop the explicit String, Number, and Boolean annotations from
                # Calflow 1.0 when roundtripping.
                return underlying_json_str
            else:
                return [OpType.Value.value, [schema, underlying_json_str]]
    else:
        raise Exception(f"Op with unknown type: {op}")


def type_args_to_lispress(type_args: List[TypeName]) -> Optional[Lispress]:
    """Converts the provided list of type args into a Lispress expression."""
    if len(type_args) == 0:
        return None
    return [type_name_to_lispress(targ) for targ in type_args]


def type_name_to_lispress(type_name: TypeName) -> Lispress:
    """Converts the provided type name into a Lispress expression."""
    if len(type_name.type_args) == 0:
        return type_name.base
    else:
        base: List[Sexp] = [type_name.base]
        type_args = [type_name_to_lispress(targ) for targ in type_name.type_args]
        return base + type_args


def _sugar_gets(sexp: Lispress) -> Lispress:
    """A sugaring that converts `(get X #(Path "y"))` to `(:y X)`.  (inverse of `unsugar_gets`)"""
    if isinstance(sexp, str) or len(sexp) == 0:
        return sexp
    else:
        hd, *tl = sexp
        if hd == DataflowFn.Get.value:
            assert (
                len(tl) == 2
            ), f"{DataflowFn.Get.value} expected 2 arguments, got {len(tl)}: {tl}"
            obj, path_value = tl
            try:
                _value, (_path, key) = path_value
                # If `key` is an alphanumeric string surrounded by double quotes,
                # strip off quotes and turn it into a (:key obj) style accessor.
                # The alphanumeric test is to make sure it doesn't have spaces or other
                # reserved symbols which would mess up the round-trip.
                if isinstance(key, str) and key.startswith('"') and key.endswith('"'):
                    key = key[1:-1]
                    if key.isalnum():
                        return [_key_to_named_arg(key), _sugar_gets(obj)]
            except ValueError:
                # if the `path` is not a simple `#(Path "some_path")` style constructor we
                # won't do the sugaring.
                return sexp
        return [_sugar_gets(s) for s in sexp]


def _desugar_gets(sexp: Lispress) -> Lispress:
    """Converts `(:y X)` to `(get X #(Path "y"))`. (inverse of `sugar_gets`)"""
    if isinstance(sexp, str) or len(sexp) == 0:
        return sexp
    else:
        hd, *tl = sexp
        if isinstance(hd, str) and _is_named_arg(hd):
            key = _named_arg_to_key(hd)
            assert (
                len(tl) == 1
            ), f"key accessor only expects 1 argument, got {len(tl)}: {tl}"
            (obj,) = tl
            return [
                DataflowFn.Get.value,
                _desugar_gets(obj),
                [OpType.Value.value, ["Path", f'"{key}"']],
            ]
        return [_desugar_gets(s) for s in sexp]


def _roots_and_reentrancies(program: Program) -> Tuple[Set[str], Set[str]]:
    ids = {e.id for e in program.expressions}
    arg_counts = Counter(a for e in program.expressions for a in e.arg_ids)
    roots = ids.difference(arg_counts)  # ids that are never used as args
    reentrancies = {
        i for i, c in arg_counts.items() if c >= 2
    }  # args that are used multiple times as args
    return roots, reentrancies


def _program_to_unsugared_lispress(program: Program) -> Lispress:
    """
    Nests `program` into an s-expression.
    Any expressions that are used more than once (i.e. reentrancies) are
    defined at the top level, as late as possible before their first use.
    The program is maximally nested, with only roots and reentrancies at the top
    level.
    """
    if len(program.expressions) == 0:
        return []

    roots, reentrancies = _roots_and_reentrancies(program)
    assert roots, "program must have at least one root"

    reentrant_ids: Dict[str, str] = {}
    sexps_by_id: Dict[str, Sexp] = {}
    root_sexps: List[Sexp] = []
    let_bindings: List[Sexp] = []
    for expression in program.expressions:
        # create a sexp for expression
        idx = expression.id
        op_lispress = op_to_lispress(expression.op)
        # if there type args, we create a META expression
        if expression.type_args is not None:
            op_type_args_lispress = type_args_to_lispress(expression.type_args)
            op_lispress = [META_CHAR, op_type_args_lispress, op_lispress]
        curr: Sexp
        if isinstance(expression.op, (BuildStructOp, CallLikeOp)):
            curr = [op_lispress]
            named_args = get_named_args(expression)
            # if all args are named (i.e., not positional), sort them alphabetically
            # TODO in principle, we could get mixed positional and names arguments,
            #   but for now that doesn't happen in SMCalFlow 2.0 so this code is good
            #   enough. This code also only works for functions with named arguments
            #   that have upper case names, which again happens to work for SMCalFlow
            #   2.0.
            has_positional = any(k is None for k, _ in named_args)
            if not has_positional:
                named_args = sorted(get_named_args(expression))  # sort alphabetically
            for arg_name, arg_id in named_args:
                if arg_name is not None and not arg_name.startswith("arg"):
                    # name of named argument
                    curr += [_key_to_named_arg(arg_name)]
                if arg_id in reentrant_ids:
                    # reentrant steps are referred to by id
                    curr += [reentrant_ids[arg_id]]
                elif arg_id in sexps_by_id:
                    # inline the sexp we already created for this arg
                    curr += [sexps_by_id[arg_id]]
                else:
                    # external reference (should not happen)
                    curr += [[EXTERNAL_LABEL, arg_id]]
        else:
            curr = op_lispress  # value
        if expression.type:
            curr = [META_CHAR, type_name_to_lispress(expression.type), curr]
        # add it to results
        if idx in reentrancies:
            # give reentrancies fresh ids as they are encountered
            new_id = _idx_to_var_str(len(reentrant_ids))
            reentrant_ids[idx] = new_id
            let_bindings.extend([new_id, curr])
        elif idx in roots:
            root_sexps += [curr]
        else:
            # otherwise save the sexp, to be inlined when it appears as an
            # argument later
            sexps_by_id[idx] = curr
    # sequence together multiple statements
    result = (
        []
        if len(root_sexps) == 0
        else root_sexps[0]
        if len(root_sexps) == 1
        else SEQUENCE_SEXP + root_sexps
    )
    # `let` bindings go at the top-level
    return [LET, let_bindings, result] if len(let_bindings) > 0 else result


_long_number_regex = re.compile("^([0-9]+)L$")


def unnest_line(
    s: Lispress, idx: Idx, var_id_bindings: Tuple[Tuple[str, int], ...],
) -> Tuple[List[Expression], Idx, Idx, Tuple[Tuple[str, int], ...]]:
    """
    Helper function for `_unsugared_lispress_to_program`.
    Converts a Lispress s-expression into a Program, keeping track of
    variable bindings and the last Expression index used.

    :param s: the Sexp to unnest
    :param idx: the highest used Expression idx so far
    :param var_id_bindings: map from linearized variable id to step id
    :return: A 4-tuple containing the resulting list of Expressions,
    the idx of this whole program,
    the most recent idx used (often the same as the idx of the program), and
    a map from variable names to their idx.
    """
    if not isinstance(s, list):
        try:
            m = _long_number_regex.match(s)
            if m is not None:
                n = m.group(1)
                expr, idx = mk_value_op(value=int(n), schema="Long", idx=idx)
                return [expr], idx, idx, var_id_bindings
            else:

                # bare value
                value = loads(s)
                known_value_types = {
                    str: "String",
                    float: "Number",
                    int: "Number",
                    bool: "Boolean",
                }
                schema = known_value_types[type(value)]
                expr, idx = mk_value_op(value=value, schema=schema, idx=idx)
                return [expr], idx, idx, var_id_bindings
        except (JSONDecodeError, KeyError):
            return unnest_line([s], idx=idx, var_id_bindings=var_id_bindings)
    elif len(s) == 0:
        expr, idx = mk_value_op(s, schema="Unit", idx=idx)
        return [expr], idx, idx, var_id_bindings
    else:
        s = [x for x in s if x != EXTERNAL_LABEL]
        hd, *tl = s
        if not isinstance(hd, str):
            if len(hd) == 3 and hd[0] == META_CHAR:
                # type args
                _meta_char, type_args, function = hd
                without_type_args = [function] + tl
                exprs, arg_idx, idx, var_id_bindings = unnest_line(
                    without_type_args, idx=idx, var_id_bindings=var_id_bindings
                )
                exprs[-1] = replace(
                    exprs[-1], type_args=[mk_type_name(targ) for targ in type_args]
                )
                return exprs, arg_idx, idx, var_id_bindings
            else:
                # we don't know how to handle this case, so we just pack the whole thing
                # into a generic value
                expr, idx = mk_value_op(value=s, schema="Object", idx=idx)
                return [expr], idx, idx, var_id_bindings
        elif _is_idx_str(hd):
            # argId pointer
            var_id_dict = dict(var_id_bindings)
            # look up step index for var
            assert hd in var_id_dict
            expr_id = var_id_dict[hd]
            return [], expr_id, idx, var_id_bindings
        elif is_express_idx_str(hd):
            # external reference
            return [], unwrap_idx_str(hd), idx, var_id_bindings
        elif hd == LET:
            assert (
                len(tl) >= 2 and len(tl[0]) % 2 == 0
            ), "let binding must have var_name, var_defn pairs and a body"
            result_exprs = []
            variables, *body_forms = tl
            for var_name, body in chunked(variables, 2):
                assert isinstance(var_name, str)
                exprs, arg_idx, idx, var_id_bindings = unnest_line(
                    body, idx, var_id_bindings
                )
                result_exprs.extend(exprs)
                var_id_bindings += ((var_name, arg_idx),)
            for body in body_forms:
                exprs, arg_idx, idx, var_id_bindings = unnest_line(
                    body, idx, var_id_bindings
                )
                result_exprs.extend(exprs)
            return result_exprs, arg_idx, idx, var_id_bindings
        elif hd == SEQUENCE:
            # handle programs that have multiple statements sequenced together
            result_exprs = []
            arg_idx = idx  # in case `tl` is empty
            for statement in tl:
                exprs, arg_idx, idx, var_id_bindings = unnest_line(
                    statement, idx, var_id_bindings
                )
                result_exprs.extend(exprs)
            return result_exprs, arg_idx, idx, var_id_bindings
        elif hd == META_CHAR:
            # type ascriptions look like (^T Expr), e.g. (^Number (+ 1 2))
            # would be (1 + 2): Number in Scala.
            # Note that there is sugar in sexp.py to parse/render `(^ T Expr)`
            # as just `^T Expr`.
            assert (
                len(tl) == 2
            ), f"Type ascriptions with ^ must have two arguments, but got {str(tl)}"
            (type_declaration, sexpr) = tl
            # Recurse on the underlying expression
            exprs, arg_idx, idx, var_id_bindings = unnest_line(
                sexpr, idx=idx, var_id_bindings=var_id_bindings
            )
            # Update is type declaration.
            exprs[-1] = replace(exprs[-1], type=mk_type_name(type_declaration))
            return exprs, arg_idx, idx, var_id_bindings
        elif hd == OpType.Value.value:
            assert (
                len(tl) >= 1 and len(tl[0]) >= 1
            ), f"Values must have format '#($schema $value)'. Found '{render_compact(s)}' instead."
            ((schema, *value_tokens),) = tl
            value = " ".join(value_tokens)
            try:
                value = loads(value)
            except JSONDecodeError:
                pass
            expr, idx = mk_value_op(value=value, schema=schema, idx=idx)
            return [expr], idx, idx, var_id_bindings
        elif is_struct_op_schema(hd):
            result = []
            kvs = []
            # not all args are named and so positional arg names are set to `None`
            pending_key = None
            for arg in tl:
                if isinstance(arg, str) and _is_named_arg(arg):
                    pending_key = arg
                else:
                    val_exprs, val_idx, idx, var_id_bindings = unnest_line(
                        arg, idx, var_id_bindings
                    )
                    result.extend(val_exprs)
                    key = None
                    if pending_key is not None:
                        key = _named_arg_to_key(pending_key)
                    kvs.append((key, val_idx))
            struct_op, idx = mk_struct_op(hd, kvs, idx)
            return result + [struct_op], idx, idx, var_id_bindings
        else:
            # CallOp
            result = []
            args = []
            for arg in tl:
                arg_exprs, arg_idx, idx, var_id_bindings = unnest_line(
                    arg, idx, var_id_bindings
                )
                result.extend(arg_exprs)
                args.append(arg_idx)
            call_op, idx = mk_call_op(hd, args=args, idx=idx)
            return result + [call_op], idx, idx, var_id_bindings


def _unsugared_lispress_to_program(fs: Lispress, idx: Idx) -> Tuple[Program, Idx]:
    arg_id_map: Tuple[Tuple[str, int], ...] = ()
    expressions = []
    if isinstance(fs, list) and len(fs) == 0:
        # special-case the empty program
        return Program(expressions=[]), idx
    elif isinstance(fs, list) and len(fs) > 0 and isinstance(fs[0], list):
        # legacy support for programs with multiple statements
        # (current format should use `SEQUENCE`)
        for f in fs:
            exprs, _, idx, arg_id_map = unnest_line(f, idx, arg_id_map)
            expressions.extend(exprs)
    else:
        exprs, _, idx, arg_id_map = unnest_line(fs, idx, arg_id_map)
        expressions.extend(exprs)
    return Program(expressions=expressions), idx
