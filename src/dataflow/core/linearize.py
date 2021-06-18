#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Tools for linearizing a program so that it's easier to predict with a seq2seq model.
"""
import re
from typing import List, Tuple

from dataflow.core.lispress import (
    LEFT_PAREN,
    META_CHAR,
    RIGHT_PAREN,
    Lispress,
    lispress_to_program,
    program_to_lispress,
    render_compact,
    strip_copy_strings,
)
from dataflow.core.program import Program
from dataflow.core.program_utils import Idx, OpType
from dataflow.core.sexp import Sexp, parse_sexp


def to_canonical_form(tokenized_lispress: str) -> str:
    """Returns canonical form of a tokenized lispress.

    The canonical form is un-tokenized and compact; it also sorts named arguments in alphabetical order.
    """
    lispress = seq_to_lispress(tokenized_lispress.split(" "))
    program, _ = lispress_to_program(lispress, 0)
    round_tripped = program_to_lispress(program)
    return render_compact(round_tripped)


def seq_to_program(seq: List[str], idx: Idx) -> Tuple[Program, Idx]:
    lispress = seq_to_lispress(seq)
    return lispress_to_program(lispress, idx)


def program_to_seq(program: Program) -> List[str]:
    lispress = program_to_lispress(program)
    return lispress_to_seq(lispress)


def lispress_to_seq(lispress: Lispress) -> List[str]:
    seq = sexp_to_seq(lispress)
    return sexp_formatter(seq)


def seq_to_lispress(seq: List[str]) -> Lispress:
    deformatted = sexp_deformatter(seq)
    return strip_copy_strings(parse_sexp(" ".join(deformatted)))


def sexp_deformatter(sexp_tokens: List[str]) -> List[str]:
    """Converts a printed s-expression back to the original s-expression.

    The inverse of sexp_formatter.
    """
    sexp_str = " ".join(sexp_tokens)

    return sexp_str.split()


def sexp_formatter(sexp_tokens: List[str]) -> List[str]:
    """The printer for an s-expression.

    Inserts spaces after/before the leading/ending double quotes for value s-expressions within the s-expression.
    """
    sexp_str = " ".join(sexp_tokens)
    # Insert spaces around the quotes in a value (use non-greedy match here)
    sexp_str = re.sub(
        rf'{OpType.Value.value} \( (\S+) "([\S+\s*]+?)" \)',
        rf'{OpType.Value.value} ( \1 " \2 " )',
        sexp_str,
    )

    # complex struct in ValueOp.value.underlying
    sexp_str = re.sub(
        r'{ "schema" : "(\S+)" , "underlying" : "([\S+\s*]+?)" }',
        r'{ "schema" : "\1" , "underlying" : " \2 " }',
        sexp_str,
    )

    return sexp_str.split()


def sexp_to_seq(s: Sexp) -> List[str]:
    if isinstance(s, list):
        if len(s) == 3 and s[0] == META_CHAR:
            (_meta, type_meta, expr) = s
            return [META_CHAR] + [y for x in [type_meta, expr] for y in sexp_to_seq(x)]
        elif len(s) == 2 and s[0] == OpType.Value.value:
            (_value, expr) = s
            return [OpType.Value.value] + [y for x in [expr] for y in sexp_to_seq(x)]
        return [LEFT_PAREN] + [y for x in s for y in sexp_to_seq(x)] + [RIGHT_PAREN]
    else:
        return [s]
